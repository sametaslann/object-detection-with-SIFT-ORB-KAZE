from random import sample, shuffle
from time import time
import os
import cv2 as cv
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset_path = "./dataset"

def load_images():
    object_dirs = os.listdir(dataset_path)
    print(object_dirs)

    dataset = {}
    dataset_train = {}
    dataset_test = {}

    for obj in object_dirs:
        dataset[obj] = os.listdir(os.path.join(dataset_path, obj))


    for obj, images in dataset.items():
        shuffle(images)
        split_point = int(0.9 * len(images))  # 90% for training
        dataset_train[obj] = images[:split_point]
        dataset_test[obj] = images[split_point:]
    
    return dataset_train, dataset_test


def apply_sift(dataset):
    sift = cv.SIFT_create(nfeatures=200, edgeThreshold=8)

    dataset_sift = {}
    print(f"Processing SIFT...")
    for obj, images in dataset.items():
        dataset_sift[obj] = []
        for img in images:
            img_path = os.path.join(dataset_path, obj, img)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            dataset_sift[obj].append((img, keypoints, descriptors ))

    return dataset_sift

def apply_orb(dataset):
    orb = cv.ORB_create(nfeatures=100, edgeThreshold=12)
    dataset_orb = {}
    print(f"Processing ORB...")
    for obj, images in dataset.items():
        dataset_orb[obj] = []
        for img_name in images:
            img_path = os.path.join(dataset_path, obj, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            keypoints, descriptors = orb.detectAndCompute(img, None)
            dataset_orb[obj].append((img, keypoints, descriptors))

    return dataset_orb

def apply_kaze(dataset):
    kaze = cv.KAZE_create(threshold=0.00001)
    dataset_kaze = {}

    print(f"Processing KAZE...")
    for obj, images in dataset.items():
        dataset_kaze[obj] = []
        for img_name in images:
            img_path = os.path.join(dataset_path, obj, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            keypoints, descriptors = kaze.detectAndCompute(img, None)
            dataset_kaze[obj].append((img, keypoints, descriptors))

    return dataset_kaze



def show_few_keypoints(data, num_instances=6, images_per_row=3, algortihm="SIFT"):
    for obj, features in data.items():
        rows = (num_instances + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
        plt.figure(figsize=(20, 5 * rows))
        plt.suptitle(f"{algortihm} Keypoints for {obj}", fontsize=20)
        
        for i in range(min(num_instances, len(features))):
            img, keypoints, _ = features[i]
            img_with_keypoints = cv.drawKeypoints(img, keypoints, None)
            
            plt.subplot(rows, images_per_row, i + 1)
            plt.imshow(img_with_keypoints, cmap='gray')
            plt.title(f"Instance {i + 1}")
            plt.axis('off')
        
        plt.show()

def recognize_objects(train_data, test_data):

    bf = None
    bf = cv.BFMatcher()
    true_labels = []
    predicted_labels = []

    for test_obj, test_features in test_data.items():
        print(f"Recognizing objects in test set for {test_obj}:")
        for _, _, test_des in test_features:
            if test_des is None:
                continue
            
            matches_count = defaultdict(int)
            for train_obj in train_data.keys():
                matches_count[train_obj] = 0
            
            for train_obj, train_features in train_data.items():
                for _, _, train_des in train_features:
                    if train_des is None:
                        continue
                    
                    matches = bf.knnMatch(test_des, train_des, k=2)
                    try:
                        good_matches = [m for m, n in matches if m.distance < 0.45 * n.distance]
                    except:
                        good_matches = []
                    
                    matches_count[train_obj] += len(good_matches)
            
            recognized_obj = max(matches_count, key=matches_count.get)
            true_labels.append(test_obj)
            predicted_labels.append(recognized_obj)
            # print(f"Test image recognized as {recognized_obj}")

    return true_labels, predicted_labels


def show_confusion_matrix(true_labels, predicted_labels, method):
    labels = sorted(list(set(true_labels)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {method.upper()}')
    plt.show()


def train_with_sift(dataset_train, dataset_test):    
    start_time = time()

    dataset_train_sift = apply_sift(dataset_train)
    dataset_test_sift = apply_sift(dataset_test)
    true_labels, predicted_labels = recognize_objects(dataset_train_sift, dataset_test_sift)

    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time of SIFT: {execution_time} seconds")


    show_confusion_matrix(true_labels, predicted_labels, method="SIFT")
    show_few_keypoints(dataset_train_sift)



def train_with_orb(dataset_train, dataset_test):    
    start_time = time()

    dataset_train_orb = apply_orb(dataset_train)
    dataset_test_orb = apply_orb(dataset_test)
    true_labels, predicted_labels = recognize_objects(dataset_train_orb, dataset_test_orb, method="ORB")
    
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time of ORB: {execution_time} seconds")


    show_confusion_matrix(true_labels, predicted_labels, method="ORB")
    show_few_keypoints(dataset_train_orb, algortihm="ORB")


def train_with_kaze(dataset_train, dataset_test):    
    start_time = time()
    
    dataset_train_kaze = apply_kaze(dataset_train)
    dataset_test_kaze = apply_kaze(dataset_test)
    true_labels, predicted_labels = recognize_objects(dataset_train_kaze, dataset_test_kaze, method="KAZE")

    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time of KAZE: {execution_time} seconds")

    show_confusion_matrix(true_labels, predicted_labels, method="KAZE")
    show_few_keypoints(dataset_train_kaze, algortihm="KAZE")


dataset_train, dataset_test = load_images()

train_with_sift(dataset_train, dataset_test)
train_with_orb(dataset_train, dataset_test)
train_with_kaze(dataset_train, dataset_test)



#appy orb
