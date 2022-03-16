import pyautogui
from lips import Lips
from eyes import Blink
from eyes import Eyes
import cv2

cam=cv2.VideoCapture(0)
lips_model = Lips(coord_on = False)
eyes_model = Eyes(sens = 1, coord_on = False)
while True:
  isTrue, frame = cam.read()
  lips_expression, mag = lips_model.detect(frame)
  blink_expression, gaze_expression = eyes_model.detect(frame)

  if lips_expression == "LEFT":
    pyautogui.hscroll(-mag)
  elif lips_expression == "RIGHT":
    pyautogui.hscroll(mag)
  elif lips_expression == "UP":
    pyautogui.scroll(mag)
  elif lips_expression == "DOWN":
    pyautogui.scroll(-mag)

  if blink_expression != "NONE":
    pyautogui.click()
  
  if not gaze_expression:
    pyautogui.moveTo(gaze_expression[0], gaze_expression[1])

  #press "q" to break the video feed
  k = cv2.waitKey(1)
  if cv2.waitKey(1) == ord("q"):
    break
cam.release()
cv2.destroyAllWindows()