# app_focus_tracker_debug.py
# Requirements:
# pip install torch torchvision torchaudio mss pygetwindow opencv-python pyautogui
# (pyautogui only required if MODE == "absolute")
#
# Usage:
# - Set TARGET_TITLE to an exact title or a substring of your app's window title.
# - MODE = "relative"   -> sends relative deltas via SendInput (works when cursor is locked)
# - MODE = "absolute"   -> uses pyautogui.moveTo to teleport cursor (requires lock OFF)
# - Run. Press 'q' in the preview window to quit.

import time
import math
import ctypes
from collections import deque

import torch
import cv2
import numpy as np
import mss
import pygetwindow as gw

# Config -----
TARGET_TITLE = "Counter-Strike 2"   # <-- change this to your app title or a substring
MODE = "relative"                     # "relative" (default, works with cursor lock) or "absolute"
RES_W, RES_H = 1280, 720               # resize frame for speed (keep small for max FPS)
YOLO_CALL_SIZE = 1280                   # internal YOLO size (lower = faster)
CONF_THRESH = 0.45
SENSITIVITY = 1.0    # how aggressively to move toward target
SMOOTHING = 0.9     # small smoothing for snappy movement (0 = no smoothing)
MAX_STEP = 1500      # clamp per-frame delta (large = allow big jumps)
SHOW_PREVIEW = True
SELECT_MODE = "largest"  # "largest" or "closest"
# -----

# optional import for absolute mode
if MODE == "absolute":
    import pyautogui

# Setup SendInput for relative moves (works when cursor locked)
PUL = ctypes.POINTER(ctypes.c_ulong)
class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]
class Input_I(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MouseInput)]
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
SendInput = ctypes.windll.user32.SendInput

def move_relative(dx, dy):
    if dx == 0 and dy == 0:
        return
    mi = MouseInput(dx=int(dx), dy=int(dy), mouseData=0,
                    dwFlags=MOUSEEVENTF_MOVE, time=0,
                    dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)))
    ii = Input_I(type=INPUT_MOUSE, mi=mi)
    SendInput(1, ctypes.pointer(ii), ctypes.sizeof(ii))

# Load YOLOv5 (same approach you used before)
print("Loading model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print("Model loaded on device:", device)

# Helper: find window (exact or substring)
def find_window(title):
    # try exact match first
    wins = gw.getWindowsWithTitle(title)
    if wins:
        return wins[0]
    # fallback: substring match
    all_titles = gw.getAllTitles()
    for t in all_titles:
        if t and title.lower() in t.lower():
            w = gw.getWindowsWithTitle(t)
            if w:
                return w[0]
    return None

win = find_window(TARGET_TITLE)
if not win:
    print("Target window not found. Here are some recent window titles (first 40):")
    for t in gw.getAllTitles()[:40]:
        print("  ", repr(t))
    raise SystemExit(1)

print("Using window:", repr(win.title))

sct = mss.mss()

# small history for optional averaging (keeps tiny stability)
history = deque(maxlen=3)
smoothed_dx = 0.0
smoothed_dy = 0.0

def pick_target(dets, frame_cx, frame_cy):
    # dets: numpy array rows x1,y1,x2,y2,conf,cls (resized-frame coords)
    persons = [r for r in dets if int(r[5]) == 0 and r[4] >= CONF_THRESH]
    if not persons:
        return None
    if SELECT_MODE == "largest":
        return max(persons, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    else:  # closest to center
        return min(persons, key=lambda r: math.hypot(((r[0]+r[2]) / 2) - frame_cx, ((r[1]+r[3]) / 2) - frame_cy))

print("Starting loop. Press 'q' in preview to quit.")
while True:
    loop_t0 = time.time()

    # re-find the window each loop in case user moved/changed it
    win = find_window(TARGET_TITLE)
    if not win:
        print("Window disappeared.")
        time.sleep(0.5)
        continue

    if win.isMinimized:
        print("Window is minimized — restore it to continue.")
        time.sleep(0.5)
        continue

    win_left, win_top, win_w, win_h = win.left, win.top, win.width, win.height

    # capture only the app window region
    bbox = {"left": win_left, "top": win_top, "width": win_w, "height": win_h}
    sct_img = sct.grab(bbox)
    img = np.array(sct_img)  # BGRA
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # resize for speed
    frame_resized = cv2.resize(frame, (RES_W, RES_H))

    # run inference (fast size)
    results = model(frame_resized, size=YOLO_CALL_SIZE)
    dets = results.xyxy[0].cpu().numpy() if len(results.xyxy) else np.array([])

    # pick target
    frame_cx, frame_cy = RES_W/2.0, RES_H/2.0
    targ = pick_target(dets, frame_cx, frame_cy)

    if targ is not None:
        x1, y1, x2, y2, conf, cls = targ
        px = (x1 + x2) / 2.0
        py = (y1 + y2) / 2.0

        # Map from resized-frame coords -> window (screen) coords
        scale_x = win_w / RES_W
        scale_y = win_h / RES_H
        target_x_win = win_left + px * scale_x
        target_y_win = win_top + py * scale_y

        # For relative mode: compute offset from window center (game reads deltas)
        center_x_win = win_left + win_w / 2.0
        center_y_win = win_top + win_h / 2.0
        raw_dx = (target_x_win - center_x_win)
        raw_dy = (target_y_win - center_y_win)

        if MODE == "relative":
            # scale by sensitivity + clamp
            step_dx = max(min(raw_dx * SENSITIVITY, MAX_STEP), -MAX_STEP)
            step_dy = max(min(raw_dy * SENSITIVITY, MAX_STEP), -MAX_STEP)

            # lightweight smoothing
            smoothed_dx = SMOOTHING * smoothed_dx + (1.0 - SMOOTHING) * step_dx
            smoothed_dy = SMOOTHING * smoothed_dy + (1.0 - SMOOTHING) * step_dy

            send_dx = int(round(smoothed_dx))
            send_dy = int(round(smoothed_dy))

            # send relative deltas
            move_relative(send_dx, send_dy)

            dbg_move = f"REL dx={send_dx} dy={send_dy}"
        else:
            # absolute: move cursor to target_x_win/target_y_win
            try:
                pyautogui.moveTo(int(round(target_x_win)), int(round(target_y_win)), duration=0)
                dbg_move = f"ABS to {int(round(target_x_win))},{int(round(target_y_win))}"
            except Exception as e:
                dbg_move = f"ABS move failed: {e}"

        debug_text = f"Found person conf={conf:.2f} | {dbg_move}"
    else:
        # decay smoothing when no target
        smoothed_dx *= 0.85
        smoothed_dy *= 0.85
        debug_text = "No person detected"

    # visualization
    if SHOW_PREVIEW:
        vis = np.squeeze(results.render())  # resized frame with boxes
        cv2.putText(vis, debug_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # draw window center crosshair for reference
        cv2.drawMarker(vis, (int(frame_cx), int(frame_cy)), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.imshow("App Focused - preview", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # print lightweight console debug (every loop)
    print(f"[{time.time():.1f}] {debug_text} | window @ ({win_left},{win_top}) size ({win_w}x{win_h}) | dets={len(dets)}")

    # tiny throttle so CPU/GPU don't spike
    time.sleep(0.001)

cv2.destroyAllWindows()
