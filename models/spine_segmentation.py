import cv2
import os
import sys
import math
import imutils
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import main
import server

class Line(object):
    '''
    Simple class that holds the information related to a line;
    i.e., the slope, y-intercept, and center point along the line
    '''

    vertical_threshold = 30

    def __init__(self, m, center, start_point, end_point):
        '''
        m: slope
        center: center point along the line (tuple)
        '''

        self.m = m
        self.center = center

        self.x0 = start_point[0]
        self.y0 = start_point[1]

        self.x1 = end_point[0]
        self.y1 = end_point[1]


    def y(self, x):
        '''
        Returns the y-value of the line at position x.
        If the line is vertical (i.e., slope is close to infinity), the y-value
        will be returned as None
        '''

        # Line is vertical
        if self.m > self.vertical_threshold:
            return None

        else:
            return self.m*x + self.b


    def x(self, y):
        '''
        Returns the x-value of the line at posiion y.
        If the line is vertical (i.e., slope is close to infinity), will always
        return the center point of the line
        '''

        # Line is vertical
        if self.m > self.vertical_threshold:
            return self.center[0]

        # Line is not vertical
        else:
            return (y - self.b)/self.m


    def get_line_coordinates(self):
        return ((self.x0, self.y0), (self.x1, self.y1))


def plot_img(img, type = "gray"):
    path = main.BASE_PATH + "/data/proc_img/" + str(np.sum(img)) + ".jpg"

    #fig = plt.figure(figsize = (16,12))
    if(type == "gray"):
        plt.imshow(img, cmap = 'gray', interpolation = 'none')
        #plt.savefig(path)
        plt.show()

    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.savefig(path)
        plt.show()
    #plt.xticks([])
    #plt.yticks([])
    

def canny_edge(img, low_th, high_th, debug = False):
    edged = cv2.Canny(img, low_th, high_th)

    if debug:
        print('Canny edge detection')
        plot_img(edged)

    return edged


def gaussian_blur(img, filter_size, sigma, debug = False):
    proc_img = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)

    if debug:
        print('Gaussian Blur')
        plot_img(proc_img)

    return proc_img


def global_thresholding(img, th_value, max_value, debug = False):
    thresh_img = cv2.threshold(img, th_value, max_value, cv2.THRESH_BINARY)

    if debug:
        print('Global Thresholding')
        plot_img(thresh_img)

    return thresh_img


def adaptive_thresholding(img, block_size, C):
    thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

    if debug:
        print('Adaptive Thresholding')
        plot_img(thresh_img)

    return thresh_img


def otsu_thresholding(img):
    high_thresh, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        print('Adaptive Thresholding')
        print('Threshold used', high_thresh)
        plot_img(thresh_img)

    return thresh_img


def connected_components(img, debug = False):
    '''
    Finds all connected components in a binary image and assigns all connections
    within a component to a unique value for that component.
    Returns the processed image, and the values of the unique components.

    '''
    levels, proc_img = cv2.connectedComponents(img, connectivity = 8)

    if debug:
        print('Find connected components, levels = ', levels)
        print(proc_img.ravel())
        plt.hist(proc_img.ravel(), levels-1, [2, levels])
        plt.savefig(main.BASE_PATH + "/data/connected_components.jpg")
        plt.show()
        plot_img(proc_img)

    return proc_img, levels


def remove_short_clusters(img, levels, th = 200, debug = False):
    hist = []

    for i in range(levels):
        hist.append(0)

    for i in range(len(img)):
        for j in range(len(img[i])):
            hist[img[i][j]] += 1

    #th = 200
    max_freq = []

    #new = np.array(img)
    new_img = np.zeros(img.shape)
    np.copyto(new_img, img)

    for i in range(1, levels):
        if(hist[i] > th):
            max_freq.append(i)
            #new_img[img == i] = 255

    for l in max_freq:
        new_img[img == l] = 255

    for i in range(len(new_img)):
        for j in range(len(new_img[i])):
            if(new_img[i][j] != 255):
                new_img[i][j] = 0

    if debug:
        print('Long clusters at min cluster size', th, 'are :', len(max_freq))
        print('Remove short clusters')
        plot_img(new_img)

    return new_img


def find_contours(img, debug = False):
    cnts = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    output = image.copy()

    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        #print(peri)
        
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        cv2.drawContours(output, [approx], -1, (0, 255, 255), 2)

    if(debug):
        print("Find contours")
        plot_img(output, type = "color")
        plt.show()


def clip_line(x1, y1, x2, y2, r, c):
    y1 = -y1
    y2 = -y2
    
    if(x2-x1 == 0):
        return (x1, 0, x2, r-1, 90)
    
    m = (y2 - y1) / (x2 - x1)
    theta = math.degrees(math.atan(m))
    
    #print(m)
    
    x1_new = x1 - (y1 / m)
    x2_new = x1 + (-r - y1) / m
    
    return int(x1_new), 0, int(x2_new), r-1, theta


def apply_hough_transform(img, image, min_votes, debug = False):
    lines = cv2.HoughLines(img.astype('uint8'), 1, np.pi/180, min_votes)
    r, c = img.shape

    output = image.copy()
    #output2 = image.copy()
    
    all_theta = []
    actual_theta = []
    points = []

    for values in lines:
        rho, theta = values[0]
        all_theta.append(math.degrees(theta))
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        x3, y3, x4, y4, t = clip_line(x1, y1, x2, y2, r, c)
        actual_theta.append(t)
        #print(x1, y1, x2, y2, x0, y0, rho)
        #print(math.degrees(theta))
        #print(t)

        if(t > 20 or t < -20):
            points.append([(x3, y3), (x4, y4), t])

            #cv2.line(output, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.line(output, (x3,y3), (x4,y4), (0,255,255), 2)
        
    if(debug):
        print("Lines detected :", lines.shape)

        plot_img(output, type = "color")

        plt.hist(all_theta, 360, [-180, 180])
        plt.show()

        plt.hist(actual_theta, 180, [-90, 90])
        plt.savefig(main.BASE_PATH + "/data/theta.jpg")

        plt.show()

    return points


def merge_lines(points, image, debug = False):
    points.sort(key = lambda point: point[0][0])
    points = points
    
    #print(points)

    pset = []

    i = 0
    while(i < len(points)):
        t = []
        
        while(i < len(points)-1):
            if(points[i+1][0][0] - points[i][0][0] <= 20):
                t.append(i)
                i += 1
            else:
                break
            
        t.append(i)
        pset.append(t)
        i += 1

    new_points = []

    for p in pset:
        
        sum_x1 = 0
        sum_x2 = 0
        
        for l in p:
            sum_x1 += points[l][0][0]
            sum_x2 += points[l][1][0]
        
        sum_x1 /= len(p)
        sum_x2 /= len(p)

        new_points.append([(int(sum_x1), points[l][0][1]), (int(sum_x2), points[l][1][1]), points[l][2]])
    

    output = image.copy()

    for p in new_points:
        cv2.line(output, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0, 255, 255), 2)

    if(debug):
        print("Merge Lines", pset)    
        print(new_points)
        plot_img(output, type = "color")

    return new_points, output


def get_book_lines(img_path, submission_id, debug = False):

    image = cv2.imread(img_path)

    #resize image
    r = 1000.0 / image.shape[1]
    dim = (1000, int(image.shape[0] * r))
 
    # perform the actual resizing of the image and show it
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    #print(img_path)
    #print(os.path.basename(img_path))
    #print(os.path.dirname(img_path))

    print(image.shape)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edged = canny_edge(gray, 50, 150, debug = debug)

    proc_img, levels = connected_components(edged, debug = debug)

    proc_img = remove_short_clusters(proc_img, levels, th = 200, debug = debug)

    points = apply_hough_transform(proc_img, image, 130, debug = debug)

    points, proc_img = merge_lines(points, image, debug = debug)

    lines = []
    for i in range(len(points)):
        start_point = points[i][0]
        end_point = points[i][1]
        
        center_x = int((start_point[0] + end_point[0]) / 2)
        center_y = int((start_point[1] + end_point[1]) / 2)

        lines.append(Line(points[i][2], (center_x, center_y), start_point, end_point))


    proc_file_path = server.get_processed_image_path_from_submission_id(submission_id)
    cv2.imwrite(proc_file_path, proc_img)

    return lines, image