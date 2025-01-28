import tensorflow as tf

# Load the .h5 model
h5_model_path = "final_model.h5"  # Replace with your .h5 model file path
model = tf.keras.models.load_model(h5_model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted .tflite model
tflite_model_path = "final_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted to TFLite and saved to {tflite_model_path}")
