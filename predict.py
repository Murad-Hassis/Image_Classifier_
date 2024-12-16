import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

# Function to preprocess the image
def process_image(image_path):
    """
    Preprocess the input image to the expected format for the model.
    """
    image = Image.open(image_path)
    image = np.asarray(image) / 255.0  # Normalize the image to [0,1]
    image = tf.image.resize(image, (224, 224)).numpy()  # Resize to 224x224
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to make predictions
def predict(image_path, model, top_k):
    """
    Predict the top K probabilities and class indices for an input image.
    """
    processed_image = process_image(image_path)
    predictions = model(processed_image)  # Run inference
    probs = tf.nn.softmax(predictions[0]).numpy()  # Apply softmax
    
    top_indices = np.argsort(probs)[-top_k:][::-1]  # Top K indices sorted
    top_probs = probs[top_indices]  # Corresponding probabilities
    return top_probs, top_indices

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict flower classes using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("model_path", type=str, help="Path to the saved model directory")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to class names")

    args = parser.parse_args()
    
    # Load the model (SavedModel format)
    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Make predictions
    print("Making predictions...")
    top_probs, top_classes = predict(args.image_path, model, args.top_k)

    # Map indices to class names if a category file is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names[str(index + 1)] for index in top_classes]
    else:
        class_labels = [str(index) for index in top_classes]

    # Print results
    print("\nTop Predictions:")
    for i in range(len(top_probs)):
        print(f"{class_labels[i]}: {top_probs[i]:.4f}")

if __name__ == "__main__":
    main()
