"""
utilities for the ORL dataset.

Assumes you've already "tar -xzvf allImages.tgz" to
have a directory of all images.
dataset site: http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
"""

import numpy as np
from collections import namedtuple
import numpy.random as random
import matplotlib.pyplot as plt

RSEED = 1000 #Using a constant seed so we always
# We're just going to do it easy, first 8 of subject is training, last 2 test...

def getFaceData():
    faces = namedtuple('Faces', ['train', 'test'])
    faces.train = namedtuple('FTrain', ['images', 'labels'])
    faces.test = namedtuple('FTest', ['images', 'labels'])

    train_images = np.zeros((320, 10304)) # 40 subjects, each image 112x92
    train_labels = np.zeros(320).astype(np.int32)
    test_images = np.zeros((80, 10304))
    test_labels = np.zeros(80).astype(np.int32)

    for x in range(320):
        imNum, subNum = 1 + x % 8, 1 + x//8
        fname = 'allImages/%d-s%d.pgm' % (imNum, subNum)
        I = plt.imread(fname).astype(float)/255.0
        train_images[x,:] = I.ravel()
        train_labels[x] = subNum

    for x in range(80):
        imNum, subNum = 9 + x % 2, 1 + x//2
        fname = 'allImages/%d-s%d.pgm' % (imNum, subNum)
        I = plt.imread(fname).astype(float)/255.0
        test_images[x,:] = I.ravel()
        test_labels[x] = subNum

    # Now randomly shuffle the data (if we maintain the seed it isn't random)
    random.seed(RSEED)
    rinds = random.choice(320, 320, replace=False)
    train_images[:] = train_images[rinds,:]
    train_labels[:] = train_labels[rinds]

    einds = random.choice(80, 80, replace=False)
    test_images[:] = test_images[einds, :]
    test_labels[:] = test_labels[einds]


    faces.train.images = train_images
    faces.train.labels = train_labels
    faces.test.images = test_images
    faces.test.labels = test_labels
    return faces

def getBatch(data, labels, batchSize):
    random.seed()
    rinds = random.choice(len(data), batchSize, replace=False)
    subdata = data[rinds]
    sublabel = labels[rinds]
    return subdata, sublabel