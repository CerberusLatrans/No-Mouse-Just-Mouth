import numpy as np
import dlib
import cv2
import math

SHOW_VIDEO = False

detector = dlib.get_frontal_face_detector()

predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

class Eyes:
  def __init__(self, sens = 1, coord_on = False):
    self.leftLeft = 36
    self.leftTop1 = 37
    self.leftTop2 = 38
    self.leftRight = 39
    self.leftBottom2 = 40
    self.leftBottom1 = 41

    self.rightLeft = 42
    self.rightTop1 = 43
    self.rightTop2 = 44
    self.rightRight = 45
    self.rightBottom2 = 46
    self.rightBottom1 = 47

    self.sens = min(2, sens)
    self.coord_on = coord_on

  def detect(self, frame):
    dets = detector(frame, 0)
    #taking the primary face detected
    if not dets:
      return "NONE"
      #return []
    face  = dets[0]
    shape = predictor(frame, face)

    LL = shape.part(self.leftLeft)
    LT1 = shape.part(self.leftTop1)
    LT2 = shape.part(self.leftTop2)
    LR = shape.part(self.leftRight)
    LB2 = shape.part(self.leftBottom2)
    LB1 = shape.part(self.leftBottom1)

    RL = shape.part(self.rightLeft)
    RT1 = shape.part(self.rightTop1)
    RT2 = shape.part(self.rightTop2)
    RR = shape.part(self.rightRight)
    RB2 = shape.part(self.rightBottom2)
    RB1 = shape.part(self.rightBottom1)

    if self.coord_on:
      return [LL, LT1, LT2, LR, LB2, LB1, RL, RT1, RT2, RR, RB2, RB1]
    else:
      return self.get_expression(LL, LT1, LT2, LR, LB2, LB1, RL, RT1, RT2, RR, RB2, RB1)
  
  def get_expression(self, ll, lt1, lt2, lr, lb2, lb1, rl, rt1, rt2, rr, rb2, rb1):
    left_ratio, left_blink = self.eye_blink(ll, lt1, lt2, lr, lb2, lb1)
    right_ratio, right_blink = self.eye_blink(rl, rt1, rt2, rr, rb2, rb1)

    if left_blink and not right_blink:
      return "LEFT" + str(left_ratio) + "   " + str(right_ratio)
    elif not left_blink and right_blink:
      return "RIGHT" + str(left_ratio) + "   " + str(right_ratio)
    elif left_blink and right_blink:
      return "BOTH" + str(left_ratio) + "   " + str(right_ratio)
    else:
      return "NONE" + str(left_ratio) + "   " + str(right_ratio)

  def eye_blink(self, l, t1, t2, r,  b2, b1):
    left_to_right = math.dist((l.x, l.y), (r.x, r.y))
    top_to_bottom1 = math.dist((t1.x, t1.y), (b1.x, b1.y))
    top_to_bottom2 = math.dist((t2.x, t2.y), (b2.x, b2.y))

    ratio = (top_to_bottom1 + top_to_bottom2) / (2 * left_to_right)
    return ratio, ratio < 0.1

if SHOW_VIDEO:
  cam=cv2.VideoCapture(0)  
  model = Eyes(coord_on = False) 
  while True:
    isTrue, frame = cam.read()
    expression = model.detect(frame)
    w, h, c = frame.shape
    cv2.putText(frame, expression, (int(w/2), int(h/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    #for point in expression:
      #cv2.circle(frame, (point.x, point.y), 1, (0, 250, 0)) 
    cv2.imshow('video', frame)
    #press "q" to break the video feed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) == ord("q"):
      break
  cam.release()
  cv2.destroyAllWindows()