############################################
# Module to  detect hands on Webcam-images #
############################################
# TODO: split module in one for tracking/detecting and one for image tasks (e.g. show img, flip, scale, ...)

# before using this module mediapipe must be installed
import mediapipe as mp
import cv2 as cv
import numpy as np


class Handtracker:
    def __init__(self, port=0):
        self.port = port
        self.cap = cv.VideoCapture(self.port)
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)

    def get_img(self, flip_h = False, flip_v = False):
        success, img = self.cap.read()
        if flip_v:
            img = cv.flip(img, 1)
        if flip_h:
            img = cv.flip(img, 0)
        return img

    def flip_img(self, img, flip_h = False, flip_v = False):
        if flip_v:
            img = cv.flip(img, 1)
        if flip_h:
            img = cv.flip(img, 0)
        return img

    def convert_img(self, img, to_rgb: bool):
        if to_rgb:
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            return cv.cvtColor(img, cv.COLOR_RGB2BGR)

    def get_landmarks(self, img,  num_hands_max=1):
        # takes BGR image and max. number of hands
        # -> returns list of hands with the according landmarks
        # to access first hand, point 3:
        #   get_landmarks(img, 2)[0].landmark[3].x or results[0].landmark[3].y (between 0 and 1)

        img = self.convert_img(img, True)
        return self.hands.process(img).multi_hand_landmarks

    def get_landmarks_coord(self, flip_v=False, flip_h=False,  img: object = None, id_hand: object = 0) -> object:
        # if img is given track in img, otherwise in webcam-image
        if img is None:
            img = self.get_img(flip_v=flip_v, flip_h=flip_h)
        landmarks = self.get_landmarks(img)
        (h, w, l) = img.shape
        if landmarks is not None:
            coordinates = []
            for loc in landmarks[id_hand].landmark:
                coordinates.append([loc.x*w, loc.y*h, 0])
            return np.array(coordinates) #, landmarks
        else:
            return None

    # Does the same as get_landmarks_coord() but different implementation
    def get_points(self):
        success, img = self.cap.read()
        img = self.convert_img(img, True)
        landmarks = self.hands.process(img).multi_hand_landmarks
        h, w, l = img.shape
        self.convert_coord(landmarks, h, w)

    # TODO test
    def draw_landmarks(self, landmarks, img, id_hand=0, connections=False):
        # draws landmarks of the given hand (with connections) into the image

        if connections:
            mp.solutions.drawing_utils.draw_landmarks(img, landmarks[id_hand], mp.solutions.hands.HAND_CONNECTIONS)
        else:
            mp.solutions.drawing_utils.draw_landmarks(img, landmarks[id_hand])

    def show_image(self, img, inloop = True):
        h, w, l = img.shape
        cv.imshow('Stream', cv.flip(cv.resize(img, (int(w / 4), int(h / 4))), 1))
        if inloop:
            cv.waitKey(1)

    def convert_coord(self, landmarks, h, w, id_hand=0):
        if landmarks:
            coordinates = []
            for loc in landmarks[id_hand].landmark:
                coordinates.append([loc.x * w, loc.y * h, 0])
            return np.array(coordinates)
        else:
            return None


'''        if landmarks:
            coordinates = []
            for loc in landmarks[0].landmark:
                coordinates.append([loc.x*w, loc.y*h, 0])
            return np.array(coordinates)
        else:
            return None'''