import cv2 as cv
import numpy as np


def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)


cap = cv.VideoCapture('F:\Code_House\CVPic\Vid\GirlRiderLow.mp4')
while True:
    isTrue, frame = cap.read()
    frame_resized = rescale(frame, 0.5)
    cv.imshow('Video', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
