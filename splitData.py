import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

IMG_SIZE = 300
training_data = []
datadir = "data/test" //directory where grabbing images

categories = ["pneumonia", "emphysema"]

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
	random.shuffle(training_data)
	x = []
	y = []
	for features, label in training_data:
		x.append(features)
		y.append(label)

	x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	pickle_out = open("pickle_x.pickle", "wb")
	pickle.dump(x, pickle_out)
	pickle_out.close()

	pickle_out = open("pickle_y.pickle", "wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()

if __name__ == "__main__":
	create_training_data()