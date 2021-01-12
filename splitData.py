import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# I used this file to create train and test data for the CNN
# Also I modified the IMG_SIZE to see if it improved the accuracy

IMG_SIZE = 300 # Change accordingly for better image
training_data = [] # Stores image array and type of image
datadir = "data/train" # Directory to search for images
categories = ["pneumonia", "emphysema"] # Types of lungs inside the directory 'data/train'

def create_training_data():
	for category in categories:
		path = os.path.join(datadir, category)
		class_num = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMEAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

	'''
	Currently the data is pneumonia images in the first half of training data then followed by
	emphysema. When training the CNN, the data should be mixed and have near the same amount of 
	each possibility for best performance.
	'''
	random.shuffle(training_data)

	x = []
	y = []

	for features, label in training_data:
		x.append(features)
		y.append(label)

	# Store the data into pickle files
	# Will use these files in trainData and testData
	x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	pickle_out = open("x.pickle", "wb")
	pickle.dump(x, pickle_out)
	pickle_out.close()

	pickle_out = open("y.pickle", "wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()

if __name__ == "__main__":
	create_training_data()

