import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn





# Global image size
IMG_SIZE = 224

# --- Grad-CAM Helper Functions ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for an image."""
    grad_model = Model(
        inputs=model.input, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = np.power(heatmap, 0.8)
    return heatmap

def generate_gradcam_overlay_from_array(img, model, last_conv_layer_name, input_size=(IMG_SIZE, IMG_SIZE)):
    """
    Given a grayscale image array, generates the Grad-CAM overlay.
    Returns an overlay image (as a NumPy array).
    """
    # Resize and normalize the input image
    img_resized = cv2.resize(img, input_size) / 255.0
    img_array = np.expand_dims(np.repeat(img_resized[..., np.newaxis], 3, axis=-1), axis=0)
    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap_resized = cv2.resize(heatmap, input_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    # Create an overlay by combining the heatmap with the original image
    overlay = cv2.addWeighted(
        cv2.cvtColor((img_resized * 255).astype("uint8"), cv2.COLOR_GRAY2RGB),
        0.6, heatmap_colored, 0.4, 0
    )
    return overlay

def predict_and_explain_from_array(img, model, last_conv_layer_name):
    """
    Given a grayscale image array and a model, performs prediction and generates a Grad-CAM overlay.
    Returns a dictionary containing the predicted class, confidence, and a base64 encoded Grad-CAM image.
    """
    # Preprocess the image for prediction
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_rgb = np.repeat(img_norm[..., np.newaxis], 3, axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0)
    
    # Predict class probabilities
    prediction = model.predict(img_batch)
    pred_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    label_str = "Benign" if pred_class == 0 else "Malignant"
    
    # Generate Grad-CAM overlay
    overlay = generate_gradcam_overlay_from_array(
        img, model, last_conv_layer_name, input_size=(IMG_SIZE, IMG_SIZE)
    )
    # Encode the overlay image to a base64 string
    _, buffer = cv2.imencode('.png', overlay)
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "predicted_class": pred_class,
        "label": label_str,
        "confidence": confidence,
        "gradcam_image": overlay_b64
    }

# --- FastAPI Application Setup ---

app = FastAPI()

# Enable CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded models
model_vgg16 = None
model_vgg19 = None
model_mobilenet = None

# Define the names of the last convolutional layers for each model
LAST_CONV_VGG16 = "block5_conv3"  # For VGG16
LAST_CONV_VGG19 = "block5_conv4"  # For VGG19
LAST_CONV_MOBILENET = "Conv_1"    # For MobileNetV2

@app.on_event("startup")
def load_models():
    """Load the three models at startup."""
    global model_vgg16, model_vgg19, model_mobilenet
    try:
        model_vgg16 = tf.keras.models.load_model("lungs_models/best_vgg16.keras")
        print("VGG16 model loaded successfully.")
    except Exception as e:
        print(f"Error loading VGG16 model: {e}")
    try:
        model_vgg19 = tf.keras.models.load_model("lungs_models/best_vgg19.keras")
        print("VGG19 model loaded successfully.")
    except Exception as e:
        print(f"Error loading VGG19 model: {e}")
    try:
        model_mobilenet = tf.keras.models.load_model("lungs_models/best_mobilenet.keras")
        print("MobileNetV2 model loaded successfully.")
    except Exception as e:
        print(f"Error loading MobileNetV2 model: {e}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file and returns predictions from VGG16, VGG19, and MobileNetV2.
    The response contains for each model: predicted class, confidence, and a Grad-CAM overlay image (base64 encoded).
    """
    contents = await file.read()
    # Decode the image from the uploaded bytes into a grayscale image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Invalid image"}
    
    results = {}
    # Run prediction for each model if it is loaded
    if model_vgg16 is not None:
        results["vgg16"] = predict_and_explain_from_array(img, model_vgg16, LAST_CONV_VGG16)
    else:
        results["vgg16"] = {"error": "Model not loaded"}
    
    if model_vgg19 is not None:
        results["vgg19"] = predict_and_explain_from_array(img, model_vgg19, LAST_CONV_VGG19)
    else:
        results["vgg19"] = {"error": "Model not loaded"}
    
    if model_mobilenet is not None:
        results["mobilenet"] = predict_and_explain_from_array(img, model_mobilenet, LAST_CONV_MOBILENET)
    else:
        results["mobilenet"] = {"error": "Model not loaded"}
    
    return results

@app.get("/")
async def root():
    return {"message": "Models API is running"}

@app.get("/saher-test")
async def saher_test():
    return "this is saher test sign language API running test"
# if __name__ == "__main__":
#     # Run the FastAPI app with uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
