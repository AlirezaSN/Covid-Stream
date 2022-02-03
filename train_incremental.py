# 2022 Copyright - Alireza Sadeghi Nasab

# imports
from creme.linear_model import LogisticRegression
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy, Precision, Recall, F1
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
metric_1 = Accuracy()
metric_2 = Precision()
metric_3 = Recall()
metric_4 = F1()

for (i, (X, y)) in enumerate(dataset):
	preds = model.predict_one(X)
	model = model.fit_one(X, y)
	metric_1 = metric_1.update(y, preds)
	metric_2 = metric_2.update(y, preds)
	metric_3 = metric_3.update(y, preds)
	metric_4 = metric_4.update(y, preds)
	print("[INFO] update {} - {}, {}, {}, {}".format(i, metric_1, metric_2, metric_3, metric_4))

print()
print("[INFO] final Accuracy - {}".format(metric_1))
print("[INFO] final Precision - {}".format(metric_2))
print("[INFO] final Recall - {}".format(metric_3))
print("[INFO] final F1 - {}".format(metric_4))