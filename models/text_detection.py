import time
import cv2
import sys
import imutils
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression

sys.path.append('..')

import main
import server

sys.path.append(main.BASE_PATH + '/models/crnn_pytorch/')


'''
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import crnn_pytorch.models.crnn as crnn


model_path = main.BASE_PATH + '/models/crnn_pytorch/data/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))
'''

def text_recognition(image):
    orig_img = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = Image.fromarray(image)

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    #image = Image.open(img_path).convert('L')
    
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_pred))
    
    #print(sim_pred)
    #plt.imshow(orig_img)
    #plt.show()
    
    return sim_pred


def load_text_detector(east, debug = False):
    
    # load the pre-trained EAST text detector
    if(debug):
        print("[INFO] loading text detector...")
    
    net = cv2.dnn.readNet(east)

    return net


def forward_pass(image, net, W, H, debug = False):
    
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
     
    # show timing information on text prediction
    if(debug):
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

    return (scores, geometry)


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue
                
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def get_bounding_boxes(orig_image, boxes, rW, rH, padding, origW, origH, submission_id, debug = False):
    results = []
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # draw the bounding box on the image
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        
        # extract the actual padded ROI
        roi = orig_image[startY:endY, startX:endX]

        if(endX - startX < endY - startY):
            roi = imutils.rotate_bound(roi, 90)
        
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")

        """
        ---------------------------------------------------
        IF YOU WANT TO USE TESSERACT FOR TEXT RECOGNITION.
        ---------------------------------------------------
        """

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(roi, config=config)


        """
        -----------------------------------------------------------------------------------------------
        IF YOU WANT TO USE CRNN DEEP LEARNING MODEL FOR TEXT RECOGNITION.
        FOR THIS, YOU HAVE TO SETUP THE FOLLOWING REPOSITORY FIRST:
        https://github.com/meijieru/crnn.pytorch

        IF YOU ARE ABLE TO SETUP THE ABOVE REPOSITORY, THEN YOU CAN USE THE FOLLOWING TEXT RECOGNITION
        BY UNCOMMENTING THE CODE AT THE TOP OF THIS FILE.
        -----------------------------------------------------------------------------------------------
        """

        #text = text_recognition(roi)
        
        
        if(debug):
            plt.imshow(roi)
            plt.show()
            print(text)
        
        texts_detected_directory = server.get_texts_detected_directory_from_submission_id(submission_id)

        cv2.imwrite(texts_detected_directory + "/" + str(np.sum(roi)) + ".jpg", roi)
        
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))


    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    return results


def draw_bounding_boxes(image, results, debug = False):
    output = image.copy()

    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        #print("OCR TEXT")
        #print("========")
        #print("{}\n".format(text))
        
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        #output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

        #cv2.putText(output, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
    # show the output image
    if(debug):
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()

    return output


def text_detection(image, net, submission_id, debug = False):
    #print(main.BASE_PATH)
    east = main.BASE_PATH + '/data/frozen_east_text_detection.pb'
    min_confidence = 0.5
    width = 768
    height = 128
    padding = 0.02

    # load the input image and grab the image dimensions
    #image = cv2.imread(img)
    #image = imutils.rotate(image, 90)

    #image = cv2.resize(image, (768, 128))

    orig = image.copy()
    (origH, origW) = image.shape[:2]
     
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)
     
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    #net = load_text_detector(east, debug)

    scores, geometry = forward_pass(image, net, W, H, debug)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
    
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = get_bounding_boxes(orig, boxes, rW, rH, padding, origW, origH, submission_id, debug)

    proc_img = draw_bounding_boxes(orig, results, debug)

    
    bounding_boxes_directory = server.get_bounding_boxes_directory_from_submission_id(submission_id)

    cv2.imwrite(bounding_boxes_directory + "/" + str(np.sum(proc_img)) + ".jpg", proc_img)

    return results


def text_detection_multi_image(spines, submission_id, debug = False):

    east = main.BASE_PATH + '/data/frozen_east_text_detection.pb'
    #print(east)
    
    net = load_text_detector(east, debug)
    texts = []

    for spine in spines:
        query_image = imutils.rotate_bound(spine.image, -90)

        result = text_detection(query_image, net, submission_id, debug = debug)
    
        texts.append(result)

    return texts
