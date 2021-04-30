###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):

    #find the image points
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    original_image = imread(imgname)      
    weight = original_image.shape[1]
    
    right_image = original_image[:, int(weight/2):weight]
    left_image = original_image[:, 0:int(weight/2)]
    right_gray_image = cvtColor(right_image, COLOR_BGR2GRAY)
    left_gray_image = cvtColor(left_image, COLOR_BGR2GRAY)
    
    is_Corners1, corners_right = findChessboardCorners(right_gray_image, (4,4), None)
    is_Corners2, corners_left = findChessboardCorners(left_gray_image, (4,4), None)
    
    image_list1 = []
    image_list2 = []
    if is_Corners1:
        exact_right_corners = cornerSubPix(right_gray_image, corners_right, (11, 11), (-1, -1), criteria)
        image_list1.append(exact_right_corners)           
        corners_right_image = drawChessboardCorners(right_image,(4,4),exact_right_corners,is_Corners1)
    
    if is_Corners2:
        exact_left_corners = cornerSubPix(left_gray_image, corners_left, (11, 11), (-1, -1), criteria)
        image_list2.append(exact_left_corners) 
        corners_left_image = drawChessboardCorners(left_image,(4,4),exact_left_corners,is_Corners2)

    #create the world points
    world_points = np.zeros((32, 3), np.float32)
    test = [20, 15, 10, 5]
    for i in range(0,4):
        #right-plane
        world_points[i][0] = 20
        world_points[i][2] = test[i]
        world_points[i+4][0] = 15
        world_points[i+4][2] = test[i]
        world_points[i+8][0] = 10
        world_points[i+8][2] = test[i]
        world_points[i+12][0] = 5
        world_points[i+12][2]  = test[i]
        
        #left-plane
        world_points[i+16][2] = 20 
        world_points[i+16][1] = test[i]
        world_points[i+20][2] = 15
        world_points[i+20][1] = test[i]
        world_points[i+24][2] = 10
        world_points[i+24][1] = test[i]
        world_points[i+28][2] = 5
        world_points[i+28][1] = test[i]
    
    world_list = [] 
    world_list.append(world_points)
  
    image_arr1 = np.array(image_list1)        
    image_arr2 = np.array(image_list2)
    image_mat1 = np.matrix(image_arr1)
    for i in range(0,len(image_mat1)):
        image_mat1[i][0] = image_mat1[i][0] + int(weight/2)
    image_mat2 = np.matrix(image_arr2)
    image_mat = np.vstack((image_mat1,image_mat2))
    image_mat_T = image_mat.T
    world_arr = np.array(world_list)
    world_mat = np.matrix(world_arr)
    world_mat_T = world_mat.T
    
    R = np.ones(32)
    
    #corrdinates in world-plane
    X_mat = np.vstack([world_mat_T, R])
    
    #corrdinates in image-plane
    x_mat = np.vstack([image_mat_T, R]) 
    
    #STEP 1 calculate A using dlm
    A = np.zeros((64,12))
    for i in range(0,32):
        A[i] = [X_mat[0,i], X_mat[1,i], X_mat[2,i], 1, 0, 0, 0, 0, -x_mat[0,i]*X_mat[0,i], -x_mat[0,i]*X_mat[1,i], -x_mat[0,i]*X_mat[2,i], -x_mat[0,i] ]
        A[i+1] = [0, 0, 0, 0, X_mat[0,i], X_mat[1,i], X_mat[2,i], 1, -x_mat[1,i]*X_mat[0,i], -x_mat[1,i]*X_mat[1,i], -x_mat[1,i]*X_mat[2,i], -x_mat[1,i] ]
    
    #STEP 2 solve Ax = 0
    U, S, V = np.linalg.svd(A)
    x = V[-1]
    x = x.reshape(3,4)
    
    #STEP 3 solve M, M = lambda * x
    lambd = 1/np.sqrt(np.square(x[2,0]) + np.square(x[2,1]) + np.square(x[2,2]))
    M = np.dot(lambd, x)
        
    m1 = M[0,0:2].T
    m2 = M[1,0:2].T
    m3 = M[2,0:2].T
    
    o_x = np.matmul(m1.T,m3)
    o_y = np.matmul(m2.T,m3)
    f_x = np.sqrt(np.matmul(m1.T,m1) - np.square(o_x))
    f_y = np.sqrt(np.matmul(m2.T,m2) - np.square(o_y))
       

    intrinsic_params = [f_x, f_y, o_x, o_y]
    
    #world points in another coordinate system
    world_points2 = np.zeros((32, 3), np.float32)
    for i in range(0,4):
        #xy-plane in left plane
        world_points2[i][0] = 20
        world_points2[i][1] = test[i]
        world_points2[i+4][0] = 15
        world_points2[i+4][1] = test[i]
        world_points2[i+8][0] = 10
        world_points2[i+8][1] = test[i]
        world_points2[i+12][0] = 5
        world_points2[i+12][1] = test[i]
        
        #yz-plane in right plane
        world_points2[i+16][1] = 20 
        world_points2[i+16][2] = test[i]
        world_points2[i+20][1] = 15
        world_points2[i+20][2] = test[i]
        world_points2[i+24][1] = 10
        world_points2[i+24][2] = test[i]
        world_points2[i+28][1] = 5
        world_points2[i+28][2] = test[i]
        
    world_list2 = []
    world_list2.append(world_points2)
    world_arr2 = np.array(world_list2)
    world_mat2 = np.matrix(world_arr2)
    world_mat_T2 = world_mat2.T
    X_mat2 = np.vstack([world_mat2.T, R])

    #STEP 1 calculate A using dlm
    A2 = np.zeros((64,12))
    for i in range(0,32):
        A2[i] = [X_mat2[0,i], X_mat2[1,i], X_mat2[2,i], 1, 0, 0, 0, 0, -x_mat[0,i]*X_mat2[0,i], -x_mat[0,i]*X_mat2[1,i], -x_mat[0,i]*X_mat2[2,i], -x_mat[0,i] ]
        A2[i+1] = [0, 0, 0, 0, X_mat2[0,i], X_mat2[1,i], X_mat2[2,i], 1, -x_mat[1,i]*X_mat2[0,i], -x_mat[1,i]*X_mat2[1,i], -x_mat[1,i]*X_mat2[2,i], -x_mat[1,i] ]
    
    #STEP 2 solve Ax = 0
    U2, S2, V2 = np.linalg.svd(A2)
    x2 = V2[-1]
    x2 = x2.reshape(3,4)
    
    #STEP 3 solve M, M = lambda * x
    lambd2 = 1/np.sqrt(np.square(x2[2,0]) + np.square(x2[2,1]) + np.square(x2[2,2]))
    M2 = np.dot(lambd, x)
            
    m1_2 = M2[0,0:2].T
    m2_2 = M2[1,0:2].T
    m3_2 = M2[2,0:2].T
    
    o_x2 = np.dot(m1_2.T,m3)
    o_y2 = np.dot(m2_2.T,m3)
    f_x2 = np.sqrt(np.dot(m1_2.T,m1_2) - np.square(o_x2))
    f_y2 = np.sqrt(np.dot(m2_2.T,m2_2) - np.square(o_y2))
    
    intrinsic_params2 =  [f_x2, f_y2, o_x2, o_y2]
    
    is_constant = (intrinsic_params == intrinsic_params2)
    
    return intrinsic_params, is_constant   
if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)

