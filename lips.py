import numpy as np
import dlib
import cv2
import math
import time

SHOW_VIDEO = False

detector = dlib.get_frontal_face_detector()

predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

class Lips:
  def __init__(self, sens = 1, coord_on = False):
    self.left = 48
    self.top = 51
    self.right = 54
    self.bottom = 57
    self.sens = min(2, sens)
    self.coord_on = coord_on

  def calibrate(self, cam, duration):
    timeout = time.time() + duration
    while time.time() < timeout:
      isTrue, frame = cam.read()

  def detect(self, frame):
    dets = detector(frame)
    #taking the primary face detected
    if not dets:
      return "NONE", 0
    face  = dets[0]
    shape = predictor(frame, face)

    leftPT = shape.part(self.left)
    rightPT = shape.part(self.right)
    topPT = shape.part(self.top)
    bottomPT = shape.part(self.bottom)

    if self.coord_on:
      return [leftPT, rightPT, topPT, bottomPT]
    else:
      return self.get_expression(leftPT, rightPT, topPT, bottomPT)
  
  def get_expression(self, left, right, top, bottom):

    height = math.dist((top.x, top.y), (bottom.x, bottom.y))
    width = math.dist((left.x, left.y), (right.x, right.y))
    #print(width, height)

    left_top_Y = abs(top.y - left.y)
    left_bottom_Y = abs(bottom.y - left.y)
    right_top_Y = abs(top.y - left.y)
    right_bottom_Y = abs(bottom.y - left.y)

    left_top_X = abs(top.x - left.x)
    left_bottom_X = abs(bottom.x - left.x)
    right_top_X = abs(top.x - left.x)
    right_bottom_X = abs(bottom.x - left.x)

    smile_scale = 5 / self.sens
    frown_scale = 7 / self.sens
    side_scale = 0 / self.sens

    if left_top_X > right_top_X + side_scale and left_bottom_X > right_bottom_X + side_scale:
      return "LEFT", self.get_magnitude(right_top_X + right_bottom_X, 1)
      #return "left: {} and {}, right: {} and {}".format(left_top_X, left_bottom_X, right_top_X, right_bottom_X)
    elif left_top_X + side_scale < right_top_X and left_bottom_X + side_scale < right_bottom_X:
      return "RIGHT", self.get_magnitude(left_top_X + left_bottom_X, 1)
      #return "left: {} and {}, right: {} and {}".format(left_top_X, left_bottom_X, right_top_X, right_bottom_X)
    elif left_top_Y + smile_scale < left_bottom_Y and right_top_Y + smile_scale < right_bottom_Y:
      return "UP", self.get_magnitude(left_bottom_Y + right_bottom_Y, 35)
      #return "left: {} and {}, right: {} and {}".format(left_top_X, left_bottom_X, right_top_X, right_bottom_X)
    elif left_top_Y > left_bottom_Y + frown_scale and right_top_Y > right_bottom_Y + frown_scale:
      return "DOWN", self.get_magnitude(left_bottom_Y + right_bottom_Y, 0)
      #return "left: {} and {}, right: {} and {}".format(left_top_X, left_bottom_X, right_top_X, right_bottom_X)
    
    else:
      return "NONE", 0
      #return "left: {} and {}, right: {} and {}".format(left_top_X, left_bottom_X, right_top_X, right_bottom_X)
    
  def get_magnitude(self, x, offset):
    return 1.08 ** (x - offset)

if SHOW_VIDEO:
  cam=cv2.VideoCapture(0)   
  model = Lips(coord_on = False)
  model.calibrate(cam, 5)
  while True:
    isTrue, frame = cam.read()
    expression, magnitude = model.detect(frame)
    w, h, c = frame.shape
    cv2.putText(frame, expression + ":  " + str(magnitude), (int(w/2), int(h/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('video', frame)
    #press "q" to break the video feed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) == ord("q"):
      break
  cam.release()
  cv2.destroyAllWindows()