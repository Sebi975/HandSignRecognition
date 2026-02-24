import handtracking as ht
import cv2 as cv
import mediapipe as mp
import pandas as pd
import numpy as np
from numpy import matmul
from math import sin, cos, sqrt, pi
from scipy.optimize import minimize
import time


def transpose(matrix):
    new_matrix = []
    new_row = []
    for column in range(0, len(matrix[0])):
        for row in matrix:
            new_row.append(row[column])
        new_matrix.append(new_row)
        new_row = []
    return np.array(new_matrix)


def diff_points(point1, point2):
    x = 0
    for i in range(len(point1)):
        x += abs(point1[i] - point2[i]) ** 2
    return sqrt(x)


def compare_on_plane(points1, points2):
    x = 0
    for line in range(0, points1.shape[0]):
        if line in [0, 5, 9, 13, 17]:
            x += diff_points(points1[line], points2[line])
    return sqrt(x)


def func_hand(parameters):
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
         [0, 0, 0]]  # <-- third number is zero because of projection onto 2D-Plane (only if stretch is last operation)
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
    for i in range(0, 21):
        new_points[i] = np.append(palm_points[i], [1])
    modified_points = transpose(matmul(s, matmul(r_z, matmul(r_y, matmul(r_x, matmul(m, transpose(new_points)))))))
    score = compare_on_plane(modified_points, points_img)
    return score


palm_points = np.empty([5, 3])
print(palm_points)
palm_points = pd.read_csv(f'Position_{1}_landmark_positions.csv', sep=',').drop(columns=['landmark_id']).values


cap = cv.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
guess = np.array([0, 0, 1, 0, 0, 0])

while True:
    success, img = cap.read()
    h, w, l = img.shape
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    landmarks = hands.process(imgRGB).multi_hand_landmarks
    points_img = ht.get_landmarks_coord(landmarks, h, w)
    if points_img is not None:
        parameters = minimize(func_hand, guess)
        string = ''
        for i in range(3, 6):
            string += f'{(parameters.x[i]/pi)*180}|'
        print(string)
    cv.waitKey(500)
