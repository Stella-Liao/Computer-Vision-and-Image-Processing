"""
Image Stitching Problem
(Due date: Nov. 9, 11:59 P.M., 2020)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    #step1:  find keypoints with SIFT point detector
    from cv2 import xfeatures2d, cvtColor,COLOR_BGR2GRAY   
    sift = xfeatures2d.SIFT_create()
    gray_left_img = cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right_img = cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    kp_left, des_left = sift.detectAndCompute(gray_left_img, None)
    kp_right, des_right = sift.detectAndCompute(gray_right_img, None)
    
    #kp_img_left = cv2.drawKeypoints(left_img, kp_left, None)
    #kp_img_right = cv2.drawKeypoints(right_img, kp_right, None)

    def compute_ED(A,B):
        """
        calculate euclidean distance matrix 'dist',
        for example, dist[i,j] refers to the distance between A[i,] and B[j,]
        Some codes in function "compute_ED" are from "https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f" 
        """
        dist_ = np.sum(A**2,axis=1)[:,np.newaxis] + np.sum(B**2,axis=1) -2 * np.dot(A,B.T)
        dist = np.sqrt(dist_)
        return dist
    
    #step2: match the keypoints 
    def match_kps(kp1, kp2, des1, des2):
        """
        find the matched pairs of keypoints'matched_kps',
        for example, in one row of matched_kps, it will contain the coordinates of one good pair of keypoints [x1, y1, x2, y2]
        """
        
        des1_idx =[]
        des2_idx =[]
        all_dist = compute_ED(des1,des2)
        n = all_dist.shape[0]
    
        for i in range(0,n):
            tmp = np.argsort(all_dist[i])
            dis1 = all_dist[i,tmp[0]]
            dis2 = all_dist[i,tmp[1]]
            if(dis1/dis2 < 0.8):
                des1_idx.append(i)
                des2_idx.append(tmp[0])   
    
        # Find the corresponding keypoint coordinates.
        coord1 = np.array([kp1[idx].pt for idx in des1_idx])
        coord2 = np.array([kp2[idx].pt for idx in des2_idx])
        
        return coord1, coord2
    src_pts, dst_pts = match_kps(kp_left, kp_right, des_left, des_right) 
    
    #step3:calculate homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    #step4: stitch images
    def stitch(img1, img2, H):
        # warp img1 to img2 with homography H and stitch them
    
        #get the corners of img1 and img2
        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]
        corners1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2) 
        corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2) 
    
        #make sure all parts of img1 will be visiable
        corners1_ = cv2.perspectiveTransform(corners1, H)
    
        #combine all corners and get the new image's corners
        corners = np.concatenate((corners1_, corners2), axis=0)   
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    
        # get the translation matrix and caluclate new Homography
        translation_mat = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
        H_ = np.dot(translation_mat,H)

        warped_img = cv2.warpPerspective(img1, H_, (xmax-xmin, ymax-ymin))
        warped_img[-ymin:h1-ymin,-xmin:w1-xmin] = img2
    
        return warped_img

    res = stitch(left_img,right_img,H)
    
    return res
    raise NotImplementedError

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg',result_img)
    
    print("Task 1 Finished!")


