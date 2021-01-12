import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

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
				