
'''
This file is developed to calibrate FRONT_VIEW_POINTS 
: Press any button on keyboard to store the coordinates and then use for calibration
: its interactive so just need to press 4 times on image for 4 points
: returns FRONT_VIEW_POINTS
'''

import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

global image_path


class localization_calibration:
    def cal_plot(self,image):    
        im = plt.imread(image)
        fig, ax = plt.subplots()
        im = ax.imshow(im, extent=[0, 416, 0, 416])
        global clicked_points
        clicked_points = []
        def key_event(event):
            print('x: {} and y: {}'.format(event.xdata, event.ydata))
            clicked_points.append([event.xdata, event.ydata])
        cid = fig.canvas.mpl_connect('key_press_event', key_event)

        plt.show()
 
        return clicked_points


    def ground_view(self,upperleft,lowerleft,upperright,lowerright,line,flag):
        def distance(upperleft,lowerleft,upperright,lowerright):
            print("distance between upper left lower left and upper right and lower right")
            print("x :",upperleft[0] - lowerleft[0] ," ",upperright[0] - lowerright[0])
            print("y :",upperleft[1] - lowerleft[1] ," ",upperright[1] - lowerright[1])


        def drawing(upperleft, left_intersection,upperright,right_intersection,line_point1,line_point2,flag):
            im = plt.imread(image_path)
            fig, ax = plt.subplots()
            im = ax.imshow(im, extent=[0, 416, 0, 416])
            x_values = [upperleft[0],left_intersection[0]]
            y_values = [upperleft[1],left_intersection[1]]
            plt.plot(x_values, y_values, 'bo', linestyle="--")
            x_values = [upperright[0],right_intersection[0]]
            y_values = [upperright[1],right_intersection[1]]
            plt.plot(x_values, y_values, 'bo', linestyle="--")
            if flag:
                x_values = [line_point1[0],line_point2[0]]
                y_values = [line_point1[1],line_point2[1]]
                plt.plot(x_values, y_values, 'bo',linestyle = '-')
            plt.show()

        def simultaneous_substitution(point_1,point_2,line):
            m1 = (point_2[1] - point_1[1])/(point_2[0] - point_1[0])
            m2 = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
            x1 = m1*point_2[0]
            x2 = m2*line[1][0]
            y1 = point_2[1]
            y2 = line[1][1]
            x = (x1 - y1 - x2 + y2) /(m1 -m2)
            y = m1*(x - point_2[0]) + point_2[1]
            print("x :",x)
            print("y :",y)
 
            return [float(x),float(y)]


        def line_intersection(point_1,point_2,line):
            m = (point_2[1] - point_1[1])/(point_2[0] - point_1[0])
           
            y = line
            x = ((y - point_1[1] )+ m*point_1[0])/m
            return [float(x),float(y)]
        
        distance(upperleft,lowerleft,upperright,lowerright)
        if flag:
            inter1 = simultaneous_substitution(upperleft,lowerleft,line)
            inter2 = simultaneous_substitution(upperright,lowerright,line)
            drawing(upperleft, inter1,upperright,inter2,inter1,inter2,True)
        else:
            inter1 = line_intersection(upperleft,lowerleft,line)
            inter2 = line_intersection(upperright,lowerright,line)
            drawing(upperleft, inter1,upperright,inter2,None,None,False)
        print("upperleft , left intersection, upperright , right intersection")
        print("this points are according to opencv axes paste this coordinates in constants FRONT_VIEW_POINTS")
        print([(upperleft[0],416 - upperleft[1]), (inter1[0],416 - inter1[1]),(upperright[0],416 - upperright[1]),(inter2[0],416 - inter2[1])])


    

#Driver code
lc = localization_calibration()
image_path = r"/home/sm/Desktop/Lane-Detection-Using-CNN/images/frame.jpg"
print("This input is for the bottom and top line :")
cali_corr = input("is camera  calibrated properly y/n :")
if cali_corr == "n":
    print("Enter only the bottom line if calibration is properly done.")
    line = input("Enter line in format [[x1,y1],[x2,y2]] : ")
    flag = True
    print(line)
else:
    print("Enter the bottom line and top line.")
    line = int(input("Enter y : "))
    flag = False
    print(line)

print("record the points as upperleft,lowerleft,upperright,lowerright")
print("Caution :  Close the window by clicking cross or it will store points on click any button on keyboard.")
key_event_points = lc.cal_plot(image_path)
print(key_event_points)
if len(key_event_points) > 3:
    lc.ground_view(key_event_points[0],key_event_points[1],key_event_points[2],key_event_points[3],line,flag)
else:
    print("Record 4 points for calibration")