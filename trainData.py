import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

'''
This file is used to train the data. I passed this file into a school GPU Chupacabra 
that processed the data and created the models.
'''

def train():
	print("[INFO] loading data...")
	x = pickle.load(open("x.pickle", "rb"))
	y = pickle.load(open("y.pickle", "rb"))

	x = x / 255.0
	y = np.array(y)

	print("[INFO] Processing data...")
	for i in range(50):
		print("[EPOCH]: ------------ ", i, " ------------")
		model = Sequential()
		model.add(Conv2D(64, (3,3), input_shape = x.shape[1:]))
		model.add(layers.Activation(activations.relu))
		model.add(MaxPooling2D(pool_size = (2,2)))

		model.add(Conv2D(64, (3,3)))
		model.add(layers.Activation(activations.relu))
		model.add(MaxPooling2D(pool_size = (2,2)))

		model.add(Conv2D(64, (3,3)))
		model.add(layers.Activation(activations.relu))
		model.add(MaxPooling2D(pool_size = (2,2)))

		model.add(Flatten())
		model.add(Dense(64))

		model.add(Dense(1))
		model.add(Activation("sigmoid"))

		model.compile(loss = "binary_crossentropy",
						optimizer = "adam",
						metrics = ['accuracy'])

		model.fit(x, y, batch_size = 32, validation_split = 0.1)

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")

if __name__ == "__main__":
	train()


	