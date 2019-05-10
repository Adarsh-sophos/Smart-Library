import csv
import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas

sys.path.append('..')
import main


def initialize_models():
    #Xception_model = applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    VGG16_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    VGG19_model = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    InceptionV3_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    #MobileNet_model = applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    ResNet50_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    models = {'VGG16_model' : VGG16_model,
              'VGG19_model' : VGG19_model,
              'InceptionV3_model' : InceptionV3_model,
              'ResNet50_model' : ResNet50_model
             }

    return models


def get_feature(metadata, model):
    print(metadata['id'], end = " ")
    
    #model = models[model_name]
        
    img_path = os.path.join(source_dir, 'images', metadata['image'])
    
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
    
    if(metadata['id'] == '1'):
        print(features_arr.shape)
    
    return {"id": metadata['id'], "features": ','.join(features_arr)}


def start_extraction():
    source = main.BASE_PATH + "/data/train_spines/image_names.tsv"
    source_dir = os.path.dirname(source)

    models = initialize_models()

    # read the source file
    data = pandas.read_csv(source, sep='\t')
    
    images = data.T.to_dict().values()
    
    model_names = ["VGG16_model", "VGG19_model", "InceptionV3_model", "ResNet50_model"]
    
    for model_name in model_names:
        
        print("Using model:", model_name)
        features = []
        
        for image in images:
            features.append(get_feature(image, model["model_name"]))
        
        print("\n")
        
        # remove empty entries
        features = filter(None, features)

        # write to a tab delimited file
        #source_filename = os.path.splitext(source)[0].split(os.sep)[-1]

        save_features_dir = main.BASE_PATH + "/data/extracted_features/" + model_name + "_features.tsv"

        with open(save_features_dir, 'w') as output:
            w = csv.DictWriter(output, fieldnames=['id', 'features'], delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(features)


def create_image_names_file():
    fp = open("image_names.tsv", "w")

    fp.write("id\timage\n")

    for i in range(1, 107):
        fp.write(str(i) + "\t" + "SP" + str(i) + ".jpg\n")

    fp.close()


