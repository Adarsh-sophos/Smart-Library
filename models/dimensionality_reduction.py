import csv
import os
import sys

import numpy
import pandas
from sklearn import (
    decomposition,
    manifold,
    pipeline,
)


def named_model(name):
    if name == 'TSNE':
        return manifold.TSNE(random_state=0)
    if name == 'PCA-TSNE':
        tsne = manifold.TSNE(
            random_state=0, perplexity=50, early_exaggeration=6.0)
        pca = decomposition.PCA(n_components=48)
        return pipeline.Pipeline([('reduce_dims', pca), ('tsne', tsne)])
    if name == 'PCA':
        return decomposition.PCA(n_components=48)
    raise ValueError('Unknown model')


def process(data, model):
    # split the comma delimited string back into a list of values
    transformed = [d.split(',') for d in data['features']]

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = numpy.asarray(transformed).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE
    vis_data = model.fit_transform(x_data)

    # convert the results into a list of dict
    results = []
    for i in range(0, len(data)):
        results.append({
            'id': data['id'][i],
            'x': vis_data[i][0],
            'y': vis_data[i][1]
        })
    return results


def write_tsv(results, output_tsv):
    # write to a tab delimited file
    with open(output_tsv, 'w') as output:
        w = csv.DictWriter(
            output, fieldnames=['id', 'x', 'y'], delimiter='\t',
            lineterminator='\n')
        w.writeheader()
        w.writerows(results)


def main():
    
    feature_files = ["InceptionV3_model_features.tsv",
                     "ResNet50_model_features.tsv",
                     "VGG16_model_features.tsv",
                     "VGG19_model_features.tsv"]
    
    model = named_model("TSNE")
    
    for feature_file in feature_files:
        # read in the data file
        data = pandas.read_csv(feature_file, sep='\t')
        #print(data.shape)
        
        results = process(data, model)

        destination_dir = os.path.dirname(feature_file)
        #print(destination_dir)
        
        source_filename = os.path.splitext(feature_file)[0].split(os.sep)[-1]
        #print(source_filename)
        
        tsv_name = os.path.join(destination_dir, '{}_tsne.tsv'.format(source_filename))
        #print(tsv_name)

        write_tsv(results, tsv_name)


def visualize_features():
    InceptionV3 = pandas.read_csv("InceptionV3_model_features_tsne.tsv", sep = "\t")

    ResNet50 = pandas.read_csv("ResNet50_model_features_tsne.tsv", sep = "\t")

    VGG16 = pandas.read_csv("VGG16_model_features_tsne.tsv", sep = "\t")

    VGG19 = pandas.read_csv("VGG19_model_features_tsne.tsv", sep = "\t")

    fig, ax = plt.subplots(2, 2, figsize=(8,8))

    #plt.subplot(2, 2, 1)
    ax[0, 0].scatter(InceptionV3['x'], InceptionV3['y'])
    ax[0, 0].set_title("InceptionV3")
    #plt.show()

    #plt.subplot(2, 2, 1)
    ax[0, 1].scatter(ResNet50['x'], ResNet50['y'])
    ax[0, 1].set_title("ResNet50")
    #plt.show()

    #plt.subplot(2, 2, 1)
    ax[1, 0].scatter(VGG16['x'], VGG16['y'])
    ax[1, 0].set_title("VGG16")
    #plt.show()

    #plt.subplot(2, 2, 1)
    ax[1, 1].scatter(VGG19['x'], VGG19['y'])
    ax[1, 1].set_title("VGG19")
    plt.show()

    fig.savefig("image_similarity_plot.jpg")


