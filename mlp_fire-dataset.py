import cv2
import glob
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras import models, layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from joblib import dump

feature_vector = []
all_labels = []

for i, address in enumerate(glob.glob("fire_dataset\*\*")):

    img = cv2.imread(address)
    img = cv2.resize(img, (32, 32))
    img = img/255
    img = img.flatten()

    feature_vector.append(img)

    label = address.split("\\")[1]
    all_labels.append(label)

    if i % 100 == 0:
        print(f"[INFO] {i} / 1000 processed")

feature_vector = np.array(feature_vector)

le = LabelEncoder()
all_labels = le.fit_transform(all_labels)
all_labels = to_categorical(all_labels)
print(all_labels)


X_train, X_test, y_train, y_test = train_test_split(feature_vector, all_labels, test_size=0.2)

net = models.Sequential([
                        layers.Dense(300, activation = "relu", input_dim = 3072),
                        layers.Dense(40, activation = "relu"),
                        layers.Dense(2, activation = "softmax")
                        ])

net.compile(optimizer = "SGD",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])

H = net.fit(X_train, y_train, batch_size = 32, validation_data = (X_test, y_test), epochs=10)

plt.plot(H.history["accuracy"], label = "train accuracy")
plt.plot(H.history["val_accuracy"], label = "train val_accuracy")
plt.plot(H.history["loss"], label = "train loss")
plt.plot(H.history["val_loss"], label = "train loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Fire Dataset Calssification")
plt.show()
