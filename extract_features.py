# 2022 Copyright - Alireza Sadeghi Nasab

# imports
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-c", "--csv", required=True,
	help="path to output CSV file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="batch size for the network")
args = vars(ap.parse_args())


print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)
bs = args["batch_size"]


imagePaths = list(paths.list_images(args["dataset"]))
random.seed(42)
random.shuffle(imagePaths)


labels = ['none' if 'non' in p else 'covid' for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)


cols = ["feat_{}".format(i) for i in range(0, 7 * 7 * 2048)]
cols = ["class"] + cols

csv = open(args["csv"], "w")
csv.write("{}\n".format(",".join(cols)))



for (b, i) in enumerate(range(0, len(imagePaths), bs)):
	
	print("[INFO] processing batch {}/{}".format(b + 1,
		int(np.ceil(len(imagePaths) / float(bs)))))
	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	
	for imagePath in batchPaths:
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)

		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		batchImages.append(image)

	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)
	features = features.reshape((features.shape[0], 7 * 7 * 2048))

	for (label, vec) in zip(batchLabels, features):
		vec = ",".join([str(v) for v in vec])
		csv.write("{},{}\n".format(label, vec))
	
csv.close()