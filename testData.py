import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
import pickle

'''
Used this file to finally compare the results of the model produced from 
train data file to the test pickle files.
'''

def testData():
	model = tf.keras.models.Sequential()
	json_file = open("model.json", "r") # 
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")

	x_test = pickle.load(open("x_test.pickle", "rb"))
	y_test = pickle.load(open("y_test.pickle", "rb"))
	y_test = np.array(y_test)

	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

	test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)

	print("Model Accuracy: ", test_acc)

if __name__ == "__main__":
	testData()


