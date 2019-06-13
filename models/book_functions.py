import io
import re
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')
import main
import server

# Google cloud
# from google.cloud import vision
# from google.cloud.vision import types



class BoundingBox(object):
    '''
    A box containing a word or multiple words.
    The BoundingBox is defined by the vertex coordinates of the box;
    from the vertex coordinate, we can get the angles of the bounding box,
    the bounding box's center position, etc.
    '''

    def __init__(self, points):
        self.x0 = points[0] # startX
        self.y0 = points[1] # startY
        self.x1 = points[2] # endX
        self.y1 = points[3] # endY

    @property
    def center(self):
        '''
        Returns the center of the bounding box object
        '''
        xc = int((self.x0 + self.x1) / 2)
        yc = int((self.y0 + self.y1) / 2)

        return xc, yc

    def get_box_coordinates(self):
        return (self.x0, self.y0, self.x1, self.y1)


class Word(object):
    '''
    Simple Word class consisting of the word"s value ('string') and the bounding box ('bounding_box')
    containing the word
    '''

    def __init__(self, string, bounding_box):
        self.string = string
        self.bounding_box = BoundingBox(bounding_box)


class Spine(object):
    '''
    Simple struct that holds the list of words on the same spine.
    Orders the words by their y-value
    Also creates a 'sentence', a string hwere instead of storing the words
    in a list, the words are stored in a single string separated by spaces.
    '''

    def __init__(self, spine_image, line_points):

        self.image = spine_image
        self.line_points = line_points
        self.words = []
        self.sentence = ""

        # Set center of spine
        self.center_x = np.array([])
        self.center_y = np.array([])

        # Store the ordered words
        #ys = np.array([word.bounding_box.center[1] for word in words])
        #ordered_words = [words[i] for i in np.argsort(ys)]
        #self.words = ordered_words


        # Format the words as a sentence
        #sentence = ''
        #for word in self.words:
        #    sentence += word.string + ' '
        #self.sentence = sentence

    def set_spine_words(self, words):
        # Store the ordered words
        xs = np.array([word.bounding_box.center[0] for word in words])
        ordered_words = [words[i] for i in np.argsort(xs)]

        for i in range(len(ordered_words)-1):
            b1_x = ordered_words[i].bounding_box.center[0]
            b1_y = ordered_words[i].bounding_box.center[1]
            b2_x = ordered_words[i+1].bounding_box.center[0]
            b2_y = ordered_words[i+1].bounding_box.center[1]

            if(np.abs(b2_x - b1_x) < 20):
                if(b1_y > b2_y):
                    #print(ordered_words[i].string, ordered_words[i+1].string)
                    ordered_words[i], ordered_words[i+1] = ordered_words[i+1], ordered_words[i]
                    #print(ordered_words[i].string, ordered_words[i+1].string)

        self.words = ordered_words

        # Format the words as a sentence
        sentence = ""
        for word in self.words:
            sentence += word.string + " "
        self.sentence = sentence

        self.center_x = np.mean(np.array([word.bounding_box.center[0] for word in words]))
        self.center_y = np.mean(np.array([word.bounding_box.center[1] for word in words]))



class Book(object):
    '''
    An object that holds all of the relevent information for a book.
    Also holds the Spine object that the book was determined from.
    '''

    def __init__(self, book_info, spine):
        '''
        book_info is a dict of information (e.g., {'title':[title]}).
        spine is the Spine object that the book info was determined from.
        '''
        # Initialize the default book_info dict and replace with values found
        # in initializer object
        self.book_info = {'title':'NONE', 'authors':'NONE', 'publisher':'NONE', 'isbn-10':'NONE', 'isbn-13':'NONE'}
        
        for key in book_info.keys():
            self.book_info[key] = book_info[key]

        # Copy spine
        self.spine = spine

        # Get center of book
        self.center_x = spine.center_x
        self.center_y = spine.center_y


    def format_raw_book_info_to_words_list(self):
        '''
        Formats the book's book_info as a list of individual tokens
        '''
        book_words_list = []
        for key in self.book_info.keys():
            book_words_list += self.book_info[key].split(' ')

        return book_words_list

    
    def format_raw_spine_words_to_words_list(self):
        '''
        Formats the book's spine.word.string tokens as a list of individual tokens
        '''

        words = [word.string for word in self.spine.words]

        return words


    def format_preprocess_book_info_to_words_list(self):
        '''
        Formats the book's book_info as a list of individual tokens, all cast
        to lower case strings
        This is usually done in preparation for calculating a similarity measure
        between strings
        '''
        book_words_list = []
        for key in self.book_info.keys():
            book_info_string = self.book_info[key].lower()
            book_words_list += book_info_string.split(' ')

        return book_words_list

    def format_preprocess_spine_words_to_words_list(self):
        '''
        Formats the book's spine.word.string tokens as a list of individual tokens,
        all cast to lower case, and with special characters removed.
        This is usually done in preparation for calculating a similarity measure
        between strings
        '''

        words = [word.string for word in self.spine.words]

        # Remove all special characters
        for i in range(len(words)):
            words[i] = ''.join(e for e in words[i] if e.isalnum())

        # Remove all empty strings
        words = [word for word in words if words != '']

        return words


    def similarity(self, list1, list2):
        '''
        '''

        book_words = self.format_preprocess_book_info_to_words_list()
        spine_words = self.format_preprocess_spine_words_to_words_list()

        return self.bag_distance_similarity


    def bag_distance_similarity(self):
        return 0

    def get_words(self):
        return [word.string for word in self.spine.words]

    def set_price(self, price):

        # Price was not calculated
        if price == None:
            self.price = 0
            self.formatted_price = '$0.00'
        else:
            self.price = float(price)/100.
            self.formatted_price = '$' + "{0:.2f}".format(self.price)




def save_spines(spines, file_path):
    '''
    Save the words along each spine to the specified file path.
    '''
    with open(file_path, 'w') as file_handle:
        writer = csv.writer(file_handle, delimiter = ',')
        for spine in spines:
            writer.writerow([word.string for word in spine.words])


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def get_spines_from_lines(image, lines, submission_id, debug = False):

    spines = []
    for i in range(len(lines)-1):
        point1, point2 = lines[i].get_line_coordinates()
        point3, point4 = lines[i+1].get_line_coordinates()

        spine = four_point_transform(image.copy(), np.array([point1, point2, point3, point4]))

        spines.append(Spine(spine, [point1, point2, point3, point4]))

        extracted_spines_directory = server.get_extracted_spines_directory_from_submission_id(submission_id)

        cv2.imwrite(extracted_spines_directory + "/spine_" + str(i) + ".jpg", spine)

        if(debug):
            plt.imshow(cv2.cvtColor(spine, cv2.COLOR_BGR2RGB))
            plt.show()

    return spines