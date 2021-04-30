"""
Some codes in the function "detect_line" are from "https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html";
Some codes in the function "detect_circle" are from "https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html"
"""


import cv2
import numpy as np

def detect_circle(img):
    
    img_coins = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,5)

    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=20,maxRadius=50)
    circles = np.uint16(np.around(circles))
    
    coins = open('results/coins.txt', 'w')
    for c in circles[0,:]:
        
        #outlines
        cv2.circle(img_coins,(c[0],c[1]),c[2],(0,255,255),3)
        
        #centers
        #cv2.circle(img,(c[0],c[1]),2,(0,0,255),3)
        
        # write parameters of those coins in coins.txt file
        c1 = str(c[0])
        c2 = str(c[1])
        c3 = str(c[2])
        line = c1 + ',' + c2 + ',' + c3
        coins.write(line)
        coins.write("\n")

    coins.close()
    
    return img_coins

def detect_diagnal_line(edges):
    diagnal_lines = cv2.HoughLines(edges,1,np.pi/45,130)
    diagnal_lines = np.sort(diagnal_lines,axis=0)
    n = int(len(diagnal_lines)/2+1)
    final_diagnal_lines = []
    
    for i in range(1,n):
        rho1 = diagnal_lines[2*i-2][0][0]
        rho2 = diagnal_lines[2*i-1][0][0]
        rho = int((rho1+rho2)/2)
        final_diagnal_lines.append([[rho,diagnal_lines[i][0][1]]])

    #find the shorest slanted line
    diagonal_theta = round(np.rad2deg(diagnal_lines[1][0][1]))
    diagnal_lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 100, 10)

    for i in range(1,len(diagnal_lines2)):
        for x1, y1, x2, y2 in diagnal_lines2[i]:
            if x1!=x2:
                dist = (x1-x2)^2+(y1-y2)^2
                if dist < 5 and dist > 3:
                    theta = round(180 + np.rad2deg(np.arctan((x1-x2)/(y2-y1))))
                    if theta == diagonal_theta:
                        A = (y1-y2)/(x1-x2)
                        B = -1
                        C = y1-A*x1
                        rho = C/np.sqrt(A*A+B*B)
                        if rho > 0:
                                final_diagnal_lines.append([[round(rho),diagnal_lines[1][0][1]]])

    final_diagnal_lines = np.array(final_diagnal_lines)
    return final_diagnal_lines

def detect_vertical_line(edges):
    lines = cv2.HoughLines(edges,1,np.pi/90,130)

    vertical_lines = []
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            if theta >3:
                vertical_lines.append(lines[i])
    vertical_lines = np.array(vertical_lines)        

    vertical_lines = np.sort(vertical_lines,axis=0)
    n = int(len(vertical_lines)/2+1)
    final_vertical_lines = []
    for i in range(1,n):
        rho1 = vertical_lines[2*i-2][0][0]
        rho2 = vertical_lines[2*i-1][0][0]
        rho = int((rho1+rho2)/2)
        final_vertical_lines.append([[rho,vertical_lines[i][0][1]]])
    
    final_vertical_lines = np.array(final_vertical_lines)
    return final_vertical_lines
    
def detect_line(img):
    img_d = img.copy()
    img_v = img.copy()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200,apertureSize = 3)
    
    diagnal_lines = detect_diagnal_line(edges) 
    blue_lines = open('results/blue_lines.txt', 'w')
    
    for i in range(len(diagnal_lines)):
        for rho,theta in diagnal_lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img_d,(x1,y1),(x2,y2),(255,0,0),2)
            bl_line = str(rho) + ',' + str(theta)
            blue_lines.write(bl_line)
            blue_lines.write("\n")
    blue_lines.close()            

    vertical_lines = detect_vertical_line(edges)
    red_lines = open('results/red_lines.txt', 'w')
    
    for i in range(len(vertical_lines )):
        for rho,theta in vertical_lines [i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img_v,(x1,y1),(x2,y2),(0,0,255),2) 
            rl_line = str(rho) + ',' + str(theta)
            red_lines.write(rl_line)
            red_lines.write("\n")
    red_lines.close() 
    
    return img_d, img_v

if __name__ == "__main__":
    
    hough_img = cv2.imread('Hough.png')
    
    #coins
    img_coins = detect_circle(hough_img)
    cv2.imwrite('results/coins.jpg',img_coins)
    
    #lines
    img_d, img_v = detect_line(hough_img)
    cv2.imwrite('results/blue_lines.jpg',img_d)
    cv2.imwrite('results/red_lines.jpg',img_v)
    
    print("Task 2 Finished!")


