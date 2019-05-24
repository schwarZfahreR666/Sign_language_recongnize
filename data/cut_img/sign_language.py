# -*- coding: utf-8 -*

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img_list = []

while(True):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.resize(gray,(640,320))
        cv2.imshow('frame',gray)
    else:
        break 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()