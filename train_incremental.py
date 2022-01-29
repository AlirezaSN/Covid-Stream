# 2022 Copyright - Alireza Sadeghi Nasab

# imports
from creme.linear_model import LogisticRegression
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy
from creme import stream
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required=True,
	help="path to features CSV file")
ap.add_argument("-n", "--cols", type=int, required=True,
	help="# of feature columns in the CSV file (excluding class column")
args = vars(ap.parse_args())


print("[INFO] building column names...")
types = {"feat_{}".format(i): float for i in range(0, args["cols"])}
types["class"] = int


dataset = stream.iter_csv(args["csv"], target="class", converters=types)

model = Pipeline(
	("scale", StandardScaler()),
	("learn", OneVsRestClassifier(classifier=LogisticRegression()))
)


print("[INFO] starting training...")
metric = Accuracy()

for (i, (X, y)) in enumerate(dataset):
	preds = model.predict_one(X)
	model = model.fit_one(X, y)
	metric = metric.update(y, preds)
	print("INFO] update {} - {}".format(i, metric))


print("[INFO] final - {}".format(metric))