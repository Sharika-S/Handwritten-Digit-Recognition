import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image containing a handwritten digit from 0 to 9.")

@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    return model


st.info("Loading the Deep Learning model. The first time may take a few minutes.")

model = load_model()

st.success("✅ Deep Learning model is ready!")

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # Convert image to grayscale
    image = ImageOps.grayscale(image)

    # Resize to MNIST size
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    # Automatically invert if needed
    if np.mean(image_array) > 0.5:
        image_array = 1.0 - image_array

    image_array = image_array.reshape(1, 28, 28)

    prediction = model.predict(image_array, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("🎯 Prediction")

    st.success(f"Predicted Digit: **{predicted_digit}**")

    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("📊 Prediction Probabilities")

    for digit, probability in enumerate(prediction[0]):
        st.write(
            f"Digit {digit}: {probability * 100:.2f}%"
        )