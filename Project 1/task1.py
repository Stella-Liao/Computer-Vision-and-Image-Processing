###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
###############

import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    a1 = alpha * np.pi/180
    b1 = beta * np.pi/180
    g1 = gamma * np.pi/180
    
    r1_1 = [[np.cos(a1), -np.sin(a1), 0], 
            [np.sin(a1), np.cos(a1), 0],
            [0, 0, 1]]
    
    r1_2 = [[1, 0, 0],
            [0, np.cos(b1), -np.sin(b1)], 
            [0, np.sin(b1), np.cos(b1)]]
    
    r1_3 = [[np.cos(g1), -np.sin(g1), 0], 
            [np.sin(g1), np.cos(g1), 0],
            [0, 0, 1],]
    
    rotMat1 = np.matmul(np.matmul(r1_3,r1_2),r1_1)
    
    a2 = (360 - alpha) * np.pi/180
    b2 = (360 - beta) * np.pi/180
    g2 = (360 - gamma) * np.pi/180
        
    r2_1 = [[np.cos(g2), -np.sin(g2), 0], 
            [np.sin(g2), np.cos(g2), 0],
            [0, 0, 1],]
    
    r2_2 = [[1, 0, 0],
            [0, np.cos(b2), -np.sin(b2)], 
            [0, np.sin(b2), np.cos(b2)]]
    
    r2_3 = [[np.cos(a2), -np.sin(a2), 0], 
            [np.sin(a2), np.cos(a2), 0],
            [0, 0, 1]]
    
    rotMat2 = np.matmul(np.matmul(r2_3,r2_2),r2_1)
    
    return rotMat1, rotMat2


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 50
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
    
