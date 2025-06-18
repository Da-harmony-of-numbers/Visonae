#!/usr/bin/env python3
"""Main program for vision-based sound and graphic generation.

Pipeline:
    input(camera) -> motion recognition(YOLOv11-pose) ->
    event signal(OSC, MIDI) -> mapping -> output(graphic, sound)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pythonosc import udp_client
import mido


# 1) Load pose model
model = YOLO("yolo11n-pose.pt")

# 2) Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

# 3) Setup OSC and MIDI
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
try:
    midi_out = mido.open_output()
except IOError:
    midi_out = None

# Prediction options
pred_kwargs = {"conf": 0.3, "device": "cpu", "verbose": False}


def extract_event(keypoints: np.ndarray):
    """Map pose keypoints to a simple event value."""
    if len(keypoints) == 0:
        return None
    person = keypoints[0]
    nose_idx, left_wrist_idx, right_wrist_idx = 0, 9, 10
    try:
        nose_y = person[nose_idx][1]
        left_y = person[left_wrist_idx][1]
        right_y = person[right_wrist_idx][1]
    except IndexError:
        return None
    if left_y < nose_y and right_y < nose_y:
        return 1
    return None


def send_event(event):
    """Send detected event over OSC and MIDI."""
    osc_client.send_message("/motion", event)
    if midi_out is not None:
        midi_out.send(mido.Message("note_on", note=60 + event))


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame. Exiting.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    results = model(rgb, **pred_kwargs)
    result = results[0] if isinstance(results, list) else results

    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None:
        event = extract_event(keypoints)
        if event is not None:
            send_event(event)

    annotated = result.plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv11-Pose", annotated_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

