###############
##1. Design the function "rectify" to  return
# fundamentalMat: should be 3x3 numpy array to indicate fundamental matrix of two image coordinates. 
# Please check your fundamental matrix using "checkFunMat". The mean error should be less than 5e-4 to get full point.
##2. Design the function "draw_epilines" to  return
# draw1: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the left image; 
# draw2: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the right image.
# See the example of epilines on the PDF.
###############
from cv2 import imread, xfeatures2d, FlannBasedMatcher, cvtColor, COLOR_RGB2BGR, line, circle, computeCorrespondEpilines
import numpy as np
from matplotlib import pyplot as plt

def rectify(pts1, pts2):
    #get the right data form to calculate
    pts1_arr = np.array(pts1)
    pts1_mat = np.matrix(pts1_arr)
    pts2_arr = np.array(pts2)
    pts2_mat = np.matrix(pts2_arr)   
    pts1_T = pts1_mat.T
    pts2_T = pts2_mat.T
    n,x = pts1_mat.shape
    R = np.ones(n)
    pts1_T = np.vstack([pts1_T,R])
    pts2_T = np.vstack([pts2_T,R])
    
    #get A
    A = np.zeros((n,9))   
    for i in range(0,n):
        A[i] = [pts1_T[0,i]*pts2_T[0,i], pts1_T[0,i]*pts2_T[1,i], pts1_T[0,i]*pts2_T[2,i],pts1_T[1,i]*pts2_T[0,i], pts1_T[1,i]*pts2_T[1,i], pts1_T[1,i]*pts2_T[2,i],pts1_T[2,i]*pts2_T[0,i], pts1_T[2,i]*pts2_T[1,i], pts1_T[2,i]*pts2_T[2,i] ]
    
    #SVD
    U,S,V = np.linalg.svd(A)
    fmat = V[-1].reshape(3,3)
    return fmat


def draw_epilines(img1, img2, pt1, pt2, fmat):
    #convert the image to BGR form
    img1 = cvtColor(img1,COLOR_RGB2BGR)
    img2 = cvtColor(img2,COLOR_RGB2BGR)
    
    #get the size of the two images to calculate the end points
    weight1 = img1.shape[1]
    weight2 = img2.shape[1]
    
    #make the float into int
    pt1_x = np.int32(pt1[0])
    pt1_y = np.int32(pt1[1])
    p1 = (pt1_x,pt1_y)
    
    pt2_x = np.int32(pt2[0])
    pt2_y = np.int32(pt2[1])
    p2 = (pt2_x,pt2_y)
    
    #create point set to calculate the epilines
    pts_1 = [[p1[0],p1[1]],[0,0]]               
    pts_1 = np.matrix(pts_1)
    
    pts_2 = [[p2[0],p2[1]],[0,0]]               
    pts_2 = np.matrix(pts_2)
    
    #calculate epilines
    lines_2 = computeCorrespondEpilines(pts_1.reshape(-1,1,2), 2,fmat)
    lines_1 = computeCorrespondEpilines(pts_2.reshape(-1,1,2), 1,fmat)
    
    #draw2
    #start point in draw 2
    x_start_2 = 0
    y_start_2 = -lines_2[0][0,2] /lines_2[0][0,1]
    pt_start_2 = (x_start_2, y_start_2)
    
    #end point in draw 2
    x_end_2 = weight2
    y_end_2 = np.int32((-lines_2[0][0,2] - lines_2[0][0,0] * x_end_2) /lines_2[0][0,1])
    pt_end_2 = (x_end_2, y_end_2)
    
    image_2 = circle(img2, p2 , 20, (0, 255, 0), -1)
    image_2 = line(img2, pt_start_2, pt_end_2, (0, 255, 0), 3) 
    draw2 = image_2
    
    #draw1
    #start point in draw 1
    x_start_1 = 0
    y_start_1 = -lines_1[0][0,2] /lines_1[0][0,1]
    pt_start_1 = (x_start_1, y_start_1)
    
    #end point in draw 1
    x_end_1 = weight1
    y_end_1 = np.int32((-lines_1[0][0,2] - lines_1[0][0,0] * x_end_1) /lines_1[0][0,1])
    pt_end_1 = (x_end_1, y_end_1)
    
    image_1 = circle(img1, p1 , 20, (0, 255, 0), -1)
    image_1 = line(img1, pt_start_1, pt_end_1, (0, 255, 0), 3) 
    draw1 = image_1
    
    return draw1, draw2

def checkFunMat(pts1, pts2, fundMat):
    N = len(pts1)
    assert len(pts1)==len(pts2)
    errors = []
    for n in range(N):
        v1 = np.array([[pts1[n][0], pts1[n][1], 1]])#size(1,3)
        v2 = np.array([[pts2[n][0]], [pts2[n][1]], [1]])#size(3,1)
        error = np.abs((v1@fundMat@v2)[0][0])
        errors.append(error)
    error = sum(errors)/len(errors)
    return error
    
if __name__ == "__main__":
    img1 = imread('rect_left.jpeg') 
    img2 = imread('rect_right.jpeg')

    # find the keypoints and descriptors with SIFT
    sift = xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters for points match
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    dis_ratio = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            good.append(m)
            dis_ratio.append(m.distance/n.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    min_idx = np.argmin(dis_ratio) 
    
    # calculate fundamental matrix and check error
    fundMat = rectify(pts1, pts2)
    error = checkFunMat(pts1, pts2, fundMat)
    print(error)
    
    # draw epipolar lines
    draw1, draw2 = draw_epilines(img1, img2, pts1[min_idx], pts2[min_idx], fundMat)
    
    # save images
    fig, ax = plt.subplots(1,2,dpi=200)
    ax=ax.flat
    ax[0].imshow(draw1)
    ax[1].imshow(draw2)
    fig.savefig('rect.png')