import cv2
import sys
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import main

import spine_segmentation as seg
import feature_extraction_using_nn as feature_ext


def test_spine_seg():
    path = main.BASE_PATH + "/data/query_images"

    image_names = sorted(os.listdir(path), key = lambda k : int(k.split('.')[0]))

    totol_spines_in_dataset = 0

    #print(file_name)

    for (i, image) in enumerate(image_names):
        print(image)

        img_path = os.path.join(path, image)
        lines, proc_img = seg.get_book_lines(img_path, "000000000", debug = False)
        totol_spines_in_dataset += len(lines) - 1

        cv2.imwrite(main.BASE_PATH + "/data/submissions/000000000/proc_img/" + str(i+1) + ".jpg", proc_img)

    print(totol_spines_in_dataset)

    return totol_spines_in_dataset

test_spine_seg()


def test_feature_extraction_accuracy_using_nn():
    models = initialize_models()

    data = pandas.read_csv("ResNet50_model_features.tsv", sep = "\t")

    transformed = [d.split(',') for d in data['features']]

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = numpy.asarray(transformed).astype('float32')
    #print(x_data.shape)

    trained = x_data.reshape((x_data.shape[0], -1))
    print(trained.shape)

    test_path = main.BASE_PATH + "/data/test_spines"

    test_files = sorted(os.listdir(test_path), key = lambda k : int(k.split('.')[0]))
    #print(test_files)

    model = models["ResNet50_model"]

    correct = 0

    for test in test_files:
            
        img_path = os.path.join(test_path, test)
        
        #print(img_path)
        #print('is file: {}'.format(img_path))
        
        img = image.load_img(img_path, target_size=(800, 150))
        x = image.img_to_array(img)
        
        # the image is now in an array of shape (3, 224, 224)
        # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)[0]

        features_arr = np.char.mod('%f', features)
        
        features_arr = numpy.asarray(features_arr).astype('float32')
        
        #print(features_arr)
        #print(trained)
        
        distances = np.sum((features_arr - trained) ** 2, axis = 1)
        min_index = np.argmin(distances)
        max_index = np.argmax(distances)

        actual = int(test.split('.')[0])
    
        if(actual == min_index + 1):
            correct += 1
        
        print(test, min_index + 1, distances[min_index], distances[max_index])

    print(correct)
