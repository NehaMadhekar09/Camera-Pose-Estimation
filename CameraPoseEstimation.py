import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

video=cv.VideoCapture("project2.avi")

if(video.isOpened()==False):
    print("Error in opening video")

def FindIntersectionsOfFourLines(d_found, theta_found):
    intersections=[]
    parallel_id=0
    for i in range(1,4):
        if abs(theta_found[i]-theta_found[0]) <= 10:
            parallel_id=i
        else:
            A = np.array([
            [np.cos(np.deg2rad(theta_found[0])), np.sin(np.deg2rad(theta_found[0]))],
            [np.cos(np.deg2rad(theta_found[i])), np.sin(np.deg2rad(theta_found[i]))]])
            b = np.array([[d_found[0]], [d_found[i]]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append((x0,y0))
    for i in range(1,4):
        if i==parallel_id:
            continue
        else:
            A = np.array([
            [np.cos(np.deg2rad(theta_found[parallel_id])), np.sin(np.deg2rad(theta_found[parallel_id]))],
            [np.cos(np.deg2rad(theta_found[i])), np.sin(np.deg2rad(theta_found[i]))]])
            b = np.array([[d_found[parallel_id]], [d_found[i]]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append((x0,y0))
    return intersections

def displayLine(image,d,theta):
    a = np.cos(np.deg2rad(theta))
    b = np.sin(np.deg2rad(theta))
    x0 = (a * d)
    y0 = (b * d)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(image, (x1,y1), (x2,y2), (255, 0, 0), 3) 
    

def FindHoughLines(edge_image,delta_theta,delta_d):
    height, width = edge_image.shape
    d_max=int(np.sqrt(height**2 + width**2))

    thetas = np.arange(0, 180, step=delta_theta)
    ds = np.arange(-d_max, d_max, step=delta_d)
    
    H=np.zeros((len(ds),len(thetas)))

    for i in range(height):
        for j in range(width):
            if not edge_image[i,j]==0:
                x=j
                y=i
                for theta_id in range(len(thetas)):
                    d=x*np.cos(np.deg2rad(thetas[theta_id]))+y*np.sin(np.deg2rad(thetas[theta_id]))
                    # calculate the difference array
                    difference_array = np.absolute(ds-d)
                    # find the index of minimum element from the array
                    d_id = difference_array.argmin()
                    H[d_id,theta_id] += 1
    
    Hc=H.copy()
    H_sorted=np.sort(Hc.ravel())[::-1]

    numbers=0
    d_found=[]
    theta_found=[]

    for i in range(len(H_sorted)):
        indices=np.where(H==H_sorted[i])
        for x, y in zip(indices[0], indices[1]):
            d=ds[x]
            theta=thetas[y]
            check=True
            for i in range(len(d_found)):
                if abs(d_found[i] - d) <= 10 and abs(theta_found[i] - theta) <= 3:
                    check=False
                    break
            if check:
                d_found.append(d)
                theta_found.append(theta)
                # displayLine(edge_image,d,theta)
                numbers+=1
            if numbers >=4:
                break
        if numbers >=4:
            break
    
    # cv.imshow('Hough',edge_image)
    return d_found,theta_found


def ComputeHomography(world_coordinates,image_coordinates):
    A=np.empty((0,9), float)
    for i in range(len(world_coordinates)):
        x_dash=image_coordinates[i][0]
        y_dash=image_coordinates[i][1]
        x=world_coordinates[i][0]
        y=world_coordinates[i][1]
        A = np.append(A, [np.array([x,y,1,0,0,0,-x_dash*x,-x_dash*y,-x_dash])], 0)
        A = np.append(A, [np.array([0,0,0,x,y,1,-y_dash*x,-y_dash*y,-y_dash])], 0)
    
    A_transpose=A.T
    Mat=A_transpose.dot(A)
    eigVal, eigVec=np.linalg.eig(Mat)
    index_smallest_eigenVal=np.argmin(eigVal)
    solution=eigVec[:,index_smallest_eigenVal]

    HomographyMat=np.array([[solution[0],solution[1],solution[2]],[solution[3],solution[4],solution[5]],[solution[6],solution[7],solution[8]]])
    return HomographyMat


def DecomposeHomography(H,K):
    A=np.linalg.inv(K).dot(H)
    A1=A[:,0]
    A2=A[:,1]
    lambda1=np.linalg.norm(A1)
    lambda2=np.linalg.norm(A2)
    mylambda=(lambda1+lambda2)/2
    r1=A1/mylambda
    
    r2=A2/mylambda
    t=A[:1]/mylambda
    r3=np.cross(r1,r2)
    rotMatrix=np.array([[r1[0],r2[0],r3[0]],
                        [r1[1],r2[1],r3[1]],
                        [r1[2],r2[2],r3[2]]])
    return rotMatrix,t


K=np.array([[1380,0,946],
            [0,1380,527],
            [0,0,1]])

frames=[]
roll=[]
pitch=[]
yaw=[]
x_trans=[]
y_trans=[]
z_trans=[]
num=0
while(video.isOpened()):
    ret, frame1=video.read()
    if ret==True:
        frame=frame1.copy()

        Gaussian = cv.GaussianBlur(frame, (11, 11), 0)
        frame_hsv = cv.cvtColor(Gaussian, cv.COLOR_BGR2HSV)

        lower_white = np.array([10,10,210])
        upper_white = np.array([172,111,255])
    
        white_mask=cv.inRange(frame_hsv, lower_white, upper_white)
      
        result = cv.bitwise_and(frame_hsv, frame_hsv, mask = white_mask)
        
        result_RGB = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        cv.imshow('Paper Detected', result_RGB)
        gray=cv.cvtColor(result_RGB, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray,100,200)
        cv.imshow('Canny Edges', edges)
        ds,thetas=FindHoughLines(edges,5,5)

        intersections=FindIntersectionsOfFourLines(ds,thetas)

        for i in range(len(intersections)):
            cv.circle(edges, intersections[i], 10, (255, 0, 0), 3)

        world_points=[(0,0),(279,0),(279,216),(0,216)]

        sortforpt1=sorted(intersections,key=lambda c:c[1])
        min_y=sortforpt1[0]

        sortforpt2=sorted(intersections,key=lambda c:c[0],reverse=True)
        max_x=sortforpt2[0]

        sortforpt3=sorted(intersections,key=lambda c:c[1],reverse=True)
        max_y=sortforpt3[0]

        sortforpt4=sorted(intersections,key=lambda c:c[0])
        min_x=sortforpt4[0]

        image_points=[min_y,max_x,max_y,min_x]

        Homo=ComputeHomography(world_points,image_points)

        RotMat,t=DecomposeHomography(Homo,K)

        R=Rotation.from_matrix(RotMat)
        roll_r, pitch_r, yaw_r=R.as_euler('xyz')

        num+=1

        frames.append(num)
        roll.append(np.degrees(roll_r))
        pitch.append(np.degrees(pitch_r))
        yaw.append(np.degrees(yaw_r))
        x_trans.append(t[0][0])
        y_trans.append(t[0][1])
        z_trans.append(t[0][2])
        
        cv.imshow('Corners', edges)
        
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv.destroyAllWindows()

# print(z_trans)
plt.plot(frames, roll, color='b', label='roll')
plt.plot(frames, pitch, color='g', label='pitch')
plt.plot(frames, yaw, color='r', label='yaw')
plt.plot(frames, x_trans, color='c', label='x')
plt.plot(frames, y_trans, color='m', label='y')
plt.plot(frames, z_trans, color='y', label='z')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Frames")
plt.ylabel("Magnitude")
plt.title("Camera Pose Estimation")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()
