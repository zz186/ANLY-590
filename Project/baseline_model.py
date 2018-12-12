import cv2
import numpy as np
from scipy.cluster.vq import *
from tqdm import tqdm
import time
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def read_training_images(img_width = 250, img_height = 250):
    df_train = pd.read_csv("labels.csv")
    images =[]
    classes =[]
    print('')
    print("Loading Training Image...")
    for f, breed in tqdm(df_train.values):
        img = cv2.imread('train/{}.jpg'.format(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        classes.append(breed)
        images.append(cv2.resize(img, (img_width, img_height)))
    return (images,classes,df_train)

def read_testing_images(img_width = 250, img_height = 250):

    df_test = pd.read_csv('sample_submission.csv')

    images = []
    print("Loading Testing Image...")
    for f in tqdm(df_test['id'].values):
        img = cv2.imread('./test/{}.jpg'.format(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        images.append(cv2.resize(img, (img_width, img_height)))
    return (images,df_test)

def train_findFeatures(images,numWords = 1000):
    des_list = []
    total_key_points = 0
    print("\nTraining: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)
        total_key_points = total_key_points + len(kp1)
        
    print("\nTraining: Stack all the descriptors...")
    my_descriptors = np.zeros([total_key_points,64])
    base_index = 0
    for descriptor in tqdm(des_list):
        length = len(descriptor)
        my_descriptors[base_index:base_index+length] = descriptor
        base_index = base_index + length

    descriptors = my_descriptors

    # Perform k-means clustering
    print("\nTraining: Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
    voc, variance = kmeans(descriptors, numWords, 1, thresh = 1e-2)

    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTraining: Applying Bag of Word")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)

    idf = np.array(np.log((1.0 * len(images) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    a = 1
    im_features = im_features * idf * a
    # im_features = preprocessing.normalize(im_features, norm='l2')

    return (im_features,voc,idf)


def test_findFeatures(images,voc, idf, numWords = 1000):
    des_list = []
    print("\nTesting: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)


    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTesting: Applying BOW...")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    test_features = im_features * idf
    # test_features = preprocessing.normalize(test_features, norm='l2')
    return test_features


def cal_Accuracy(predict, test):
    if len(predict) != len(test):
        return -1;

    same = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            same = same + 1

    return same/len(predict)

def baseline():
    width = 200
    height = 200
    numberOfWords = 100
    (images, classes, df_train) = read_training_images(img_width=width, img_height=height)

    # split to train and test
    train_images, test_images, train_classes, test_classes = train_test_split(images, classes, test_size=0.05)
    # X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
    # y is classes

    (train_features, voc, idf) = train_findFeatures(train_images, numWords=numberOfWords)
    test_features = test_findFeatures(test_images, voc,idf, numWords=numberOfWords)

    print("==========")
    print("train_features.shape:",train_features.shape)
    print("==========")
    print("test_features.shape:", test_features.shape)
    print("==========")

    rf = RandomForestClassifier(n_estimators=500,
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     # max_features="sqrt",
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=1,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None)
    

    rf.fit(train_features,train_classes)

    predict = rf.predict(test_features)

    print("scores:",rf.score(test_features,test_classes))

    acc = cal_Accuracy(predict,test_classes)
    print("accuracy is", acc)
    
if __name__ == '__main__':
    baseline()