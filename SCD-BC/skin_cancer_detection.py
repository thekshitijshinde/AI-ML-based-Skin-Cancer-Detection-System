# # The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# # reference: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/discussion/183083
# # Data: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
# # https://keras.io/api/models/sequential/
# # https://keras.io/api/layers/core_layers/dense/
# # https://keras.io/api/layers/merging_layers/add/
# # https://keras.io/api/layers/convolution_layers/convolution2d
# # https://keras.io/api/layers/convolution_layers/convolution2d
# # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization



# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
# from tensorflow.keras.models import Sequential

# classes = {
#     0: ("actinic keratoses and intraepithelial carcinomae(Cancer)"),
#     1: ("basal cell carcinoma(Cancer)"),
#     2: ("benign keratosis-like lesions(Non-Cancerous)"),
#     3: ("dermatofibroma(Non-Cancerous)"),
#     4: ("melanocytic nevi(Non-Cancerous)"),
#     5: ("pyogenic granulomas and hemorrhage(Can lead to cancer)"),
#     6: ("melanoma(Cancer)"),
# }


# model = Sequential()
# model.add(
#     Conv2D(
#         16,
#         kernel_size=(3, 3),
#         input_shape=(28, 28, 3),
#         activation="relu",
#         padding="same",
#     )
# )
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
# model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
# model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
# model.add(Flatten())
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(Dense(256, activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(Dense(128, activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(64, activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(Dense(32, activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(7, activation="softmax"))
# model.summary()
# model.load_weights("best_model.h5")


# from flask import Flask, render_template, request
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.metrics import AUC
# import numpy as np

# app = Flask(__name__)

# dependencies = {
#     'auc_roc': AUC
# }

# verbose_name = {
#     0: "Actinic keratoses and intraepithelial carcinomae (Cancer)",
#     1: "Basal cell carcinoma (Cancer)",
#     2: "Benign keratosis-like lesions (Non-Cancerous)",
#     3: "Dermatofibroma (Non-Cancerous)",
#     4: "Melanocytic nevi (Non-Cancerous)",
#     5: "Pyogenic granulomas and hemorrhage (Can lead to cancer)",
#     6: "Melanoma (Cancer)",
# }

# # Load the trained model (Ensure the correct path)
# model = load_model('skin.h5', compile=False)

# print(model.summary())


# def predict_label(img_path):
#     test_image = image.load_img(img_path, target_size=(128, 128))  # Changed from (28,28) to (128,128)
#     test_image = image.img_to_array(test_image) / 255.0  # Normalize image
#     test_image = test_image.reshape(1, 128, 128, 3)

#     predict_x = model.predict(test_image)
#     classes_x = np.argmax(predict_x, axis=1)
    
#     return verbose_name[classes_x[0]]

# @app.route("/")
# @app.route("/first")
# def first():
#     return render_template('first.html')
    
# @app.route("/login")
# def login():
#     return render_template('login.html')   
    
# @app.route("/index", methods=['GET', 'POST'])
# def index():
#     return render_template("index.html")

# @app.route("/submit", methods=['GET', 'POST'])
# def get_output():
#     if request.method == 'POST':
#         img = request.files['my_image']
#         img_path = "D:/Skin Cancer Prediction/SOURCE CODE/static/tests/" + img.filename    
#         img.save(img_path)
        
#         predict_result = predict_label(img_path)
    
#     return render_template("prediction.html", prediction=predict_result, img_path=img_path)

# if __name__ == '__main__':
#     app.run(debug=True)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

# Class Labels
classes = {
    0: "actinic keratoses and intraepithelial carcinomae (Cancer)",
    1: "basal cell carcinoma (Cancer)",
    2: "benign keratosis-like lesions (Non-Cancerous)",
    3: "dermatofibroma (Non-Cancerous)",
    4: "melanocytic nevi (Non-Cancerous)",
    5: "pyogenic granulomas and hemorrhage (Can lead to cancer)",
    6: "melanoma (Cancer)",
}

# Define Model
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),
    Conv2D(256, kernel_size=(3, 3), activation="relu"),

    Flatten(),
    Dropout(0.2),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(7, activation="softmax")
])

# Load Weights
try:
    model.load_weights("skin.h5")
    print("Model weights loaded successfully from 'skin.h5'.")
except Exception as e:
    print(f"Error loading model weights: {e}")

# Print Model Summary
# model.summary()
