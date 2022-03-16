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
      if self.coord_on:
        return [], (200, 200), (200, 200)
      else:
        return "NONE"
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

    left_eye_img = frame[LT1.y:LB2.y, LL.x:LR.x]
    _, left_eye_img = cv2.threshold(left_eye_img, 42, 255, cv2.THRESH_BINARY)
    left_eye_img = cv2.erode(left_eye_img, None, iterations=2) #1
    left_eye_img = cv2.dilate(left_eye_img, None, iterations=4) #2
    left_eye_img = cv2.medianBlur(left_eye_img, 5) #3
    left_gray = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)

    right_eye_img = frame[RT1.y:RB2.y, RL.x:RR.x]
    _, right_eye_img = cv2.threshold(right_eye_img, 42, 255, cv2.THRESH_BINARY)
    right_eye_img = cv2.erode(right_eye_img, None, iterations=2) #1
    right_eye_img = cv2.dilate(right_eye_img, None, iterations=4) #2
    right_eye_img = cv2.medianBlur(right_eye_img, 5) #3
    right_gray = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)

    left_gaze, right_gaze = self.detect_gaze(left_gray, right_gray)

    if self.coord_on:
      return [LL, LT1, LT2, LR, LB2, LB1, RL, RT1, RT2, RR, RB2, RB1], left_gaze, right_gaze
    else:
      return self.get_expression(LL, LT1, LT2, LR, LB2, LB1, RL, RT1, RT2, RR, RB2, RB1), left_gray, right_gray
  
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
      return "NONE"# + str(left_ratio) + "   " + str(right_ratio)

  def eye_blink(self, l, t1, t2, r,  b2, b1):
    left_to_right = math.dist((l.x, l.y), (r.x, r.y))
    top_to_bottom1 = math.dist((t1.x, t1.y), (b1.x, b1.y))
    top_to_bottom2 = math.dist((t2.x, t2.y), (b2.x, b2.y))

    ratio = (top_to_bottom1 + top_to_bottom2) / (2.0 * left_to_right)
    return ratio, ratio < 0.2

  def detect_gaze(self, left_eye, right_eye):
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector = cv2.SimpleBlobDetector_create(detector_params)

    left_keypoints = detector.detect(left_eye)
    left_Xs = []
    left_Ys = []

    for point in left_keypoints:
      (x, y) = point.pt
      left_Xs.append(x)
      left_Ys.append(y)
    
    if not left_Xs:
      left_avg_x = 0  
    else:
      left_avg_x = sum(left_Xs) / len(left_Xs)

    if not left_Xs:
      left_avg_y = 0 
    else:
      left_avg_y = sum(left_Ys) / len(left_Ys)
    left_coord = (int(left_avg_x)+200, int(left_avg_y)+200)

    right_keypoints = detector.detect(right_eye)
    right_Xs = []
    right_Ys = []

    for point in right_keypoints:
      (x, y) = point.pt
      right_Xs.append(x)
      right_Ys.append(y)
    
    if not right_Xs:
      right_avg_x = 0
    else:
      right_avg_x = sum(right_Xs) / len(right_Xs)
    if not right_Ys:
      right_avg_y = 0
    else:
      right_avg_y = sum(right_Ys) / len(right_Ys)
    right_coord = (int(right_avg_x)+200, int(right_avg_y)+200)

    return left_coord, right_coord

  def get_mouse_coords(self):
    pass

if SHOW_VIDEO:
  cam=cv2.VideoCapture(0)  
  model = Eyes(coord_on = True)
  while True:
    isTrue, frame = cam.read()
    expression, left_gaze, right_gaze = model.detect(frame)
    w, h, c = frame.shape
    if model.coord_on:
      for point in expression:
        cv2.circle(frame, (point.x, point.y), 1, (0, 255, 0))
      
      cv2.circle(frame, left_gaze, 20, (255, 0 ,0))
      cv2.circle(frame, right_gaze, 20, (0, 0, 255))
      cv2.putText(frame, str(left_gaze), (int(w/2), int(h/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
      cv2.putText(frame, str(right_gaze), (int(w/2) + 100, int(h/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
      
    else:
      if expression != "NONE":
        cv2.putText(frame, expression, (int(w/2), int(h/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
      frame = right_gaze
    
    
    cv2.imshow('video', frame)
    #press "q" to break the video feed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) == ord("q"):
      break
  cam.release()
  cv2.destroyAllWindows()