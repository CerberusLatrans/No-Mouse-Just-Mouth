import pyautogui
from lips import Lips
from eyes import Eyes
import cv2

cam=cv2.VideoCapture(0)
lips_model = Lips(coord_on = False)
while True:
  isTrue, frame = cam.read()
  lips_expression = lips_model.detect(frame)
  if lips_expression == "LEFT":
    pyautogui.hscroll(-10)
  elif lips_expression == "RIGHT":
    pyautogui.hscroll(10)
  elif lips_expression == "UP":
    pyautogui.scroll(1)
  elif lips_expression == "DOWN":
    pyautogui.scroll(-1)
  #press "q" to break the video feed
  k = cv2.waitKey(1)
  if cv2.waitKey(1) == ord("q"):
    break
cam.release()
cv2.destroyAllWindows()