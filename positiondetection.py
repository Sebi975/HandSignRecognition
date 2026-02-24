###########################################
# Module to detect the position of a hand #
###########################################
# TODO: make it possible detect on single pictures (see: handtracking-mudule)

import handtracking as ht
from handtracking import Handtracker
import cv2 as cv
import mediapipe as mp
import pandas as pd
import numpy as np
from numpy import matmul, transpose
from math import sin, cos, sqrt, pi
from scipy.optimize import basinhopping, minimize, Bounds
import time

new_version = False


def diff_points(point1, point2):
    #x = 0
    #for i in range(len(point1)):
     #   x += (point1[i] - point2[i]) ** 2
      #  print(point1[i] - point2[i])

    return np.sqrt(np.sum((point1-point2)**2))


def compare_on_plane(points1, points2):
    x = 0
    for i in range(0, 21):
        x += diff_points(points1[i], points2[i]) ** 2
    return sqrt(x)


class PositionDetection:
    def __init__(self, port, pos_amount):
        self.modified_points = None
        self.points_img = None
        self.parameters = None
        self.position_guess = None
        self.amount_of_positions = pos_amount
        self.guess = np.array([0, 0, 1, pi, 0, 0])
        self.hand_model_points = []
        self.handtracker = Handtracker(port)
        self.position_id = 0
        self.score_list = []
        self.parameter_list = []

        ########## LOAD DATA #########
        for i in range(0, self.amount_of_positions):
            try:
                temp = pd.read_csv(f'data\Position_{i}_landmark_positions.csv', sep=',')
                self.hand_model_points.append(temp.drop(columns=['landmark_id']).values)
            except:
                print('Could not find all position files!')

    # TODO possible improvement: include position id in parameters as int with bounds
    def func_hand(self, parameters):
        m_x = parameters[0]
        m_y = parameters[1]
        s_x = s_y = parameters[2]
        alpha = parameters[3]
        beta = parameters[4]
        gamma = parameters[5]
        m = [[1, 0, 0, m_x],
             [0, 1, 0, m_y],
             [0, 0, 1, 0]]
        s = [[s_x, 0, 0],
             [0, s_y, 0],
             [0, 0, 0]]  # <-- third number is zero because of projection onto 2D-Plane (only if stretch=last operation)
        r_x = [[1, 0, 0],
               [0, cos(alpha), -sin(alpha)],
               [0, sin(alpha), cos(alpha)]]
        r_y = [[cos(beta), 0, sin(beta)],
               [0, 1, 0],
               [-sin(beta), 0, cos(beta)]]
        r_z = [[cos(gamma), -sin(gamma), 0],
               [sin(gamma), cos(gamma), 0],
               [0, 0, 1]]
        new_points = np.zeros((21, 4))
        # print(transpose(hand_model_points[position_id]))
        # print(np.ones((1,21)))
        new_points = np.append(transpose(self.hand_model_points[self.position_id]), np.ones((1, 21)), axis=0)
        # for i in range(0, 21):
        #     new_points[i] = np.append(hand_model_points[position_id][i], [1])
        # print(new_points)

        self.modified_points = transpose(matmul(s, matmul(r_z, matmul(r_y, matmul(r_x, matmul(m, new_points))))))
        # print(f'mod points:{self.modified_points}')
        # print(f'img points:{self.points_img}'
        # print(self.modified_points)
        # print(self.points_img)
        score = compare_on_plane(self.modified_points, self.points_img)
        return score

    def detect_position(self):

        self.points_img = self.handtracker.get_landmarks_coord(flip_v=True)

        if self.points_img is not None:
            self.score_list = []
            self.parameter_list = []
            for i, pos in enumerate(self.hand_model_points):
                self.position_id = i
                self.parameters = minimize(self.func_hand, self.guess)  # , bounds=((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
                # print(self.parameters.x)
                self.score_list.append(self.parameters.fun)
                self.parameter_list.append(self.parameters.x)
            self.position_guess = self.score_list.index(min(self.score_list))
            # self.guess = self.parameter_list[self.position_guess]
            # print(f'Position: {position_guess}', end=' ')
            '''
            string = ''
            for i in range(3, 6):
                string += f'{(self.parameter_list[self.score_list.index(min(self.score_list))][i] / pi) * 180}|'
            print(f'Rotation: {string}')'''
            return self.position_guess
        else:
            return None


'''    if new_version:
        def detect_position(self):
            # print(self.points_img)
            try:
                img = self.handtracker.get_img()
                self.points_img = self.handtracker.get_landmarks_coord(flip_v=True, img=img)
                if self.points_img is not None:

                    score_list_0 = []
                    score_list_1 = []

                    self.parameter_list = []
                    print('Right Hand')

                    for i, pos in enumerate(self.hand_model_points):
                        self.position_id = i
                        self.parameters = minimize(self.func_hand,
                                                   self.guess)  # , bounds=((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
                        # print(self.parameters.x)
                        score_list_0.append(
                            self.func_hand(self.parameters.x))  # look if minimize also return minimum value
                        # self.parameter_list.append(self.parameters.x)

                    print('Left Hand')
                    self.points_img = self.handtracker.get_landmarks_coord(flip_v=False, img=img)
                    # print(self.points_img)
                    for j, pos in enumerate(self.hand_model_points):
                        self.position_id = j
                        self.parameters = minimize(self.func_hand,
                                                   self.guess)  # , bounds=((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
                        # print(self.parameters.x)
                        score_list_1.append(self.func_hand(self.parameters.x))
                        # self.parameter_list.append(self.parameters.x)
                    if min(score_list_0) < min(score_list_1):
                        self.position_guess = score_list_0.index(min(score_list_0))
                    else:
                        self.position_guess = score_list_1.index(min(score_list_1))
                    # self.guess = self.parameter_list[self.position_guess]
                    # print(f'Position: {position_guess}', end=' ')
                    
                    string = ''
                    for i in range(3, 6):
                        string += f'{(self.parameter_list[self.score_list.index(min(self.score_list))][i] / pi) * 180}|'
                    print(f'Rotation: {string}')
                    return self.position_guess
                else:
                    return None
            except():
                print('failed to detect')
    else:
'''
