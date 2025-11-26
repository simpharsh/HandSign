import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

# --- Configuration ---
MODEL_PATH = "best_model.h5"  # Or "asl_model.tflite"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5  # Number of frames to confirm a prediction
SPEECH_COOLDOWN = 3.0 # Seconds between speaking the same letter/word

# Classes from the dataset
CLASSES = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'del', 'nothing', 'space'])

def init_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        return engine
    except Exception as e:
        print(f"Warning: Could not initialize TTS engine: {e}")
        return None

def speak(engine, text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

# --- Fix for DepthwiseConv2D 'groups' error ---
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

def main():
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup TTS
    engine = init_engine()
    last_speech_time = 0
    
    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    
    # Variables for smoothing
    prediction_history = []
    current_text = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Define ROI (Region of Interest) - usually a square in the center or side
        # Let's use a square box 
        box_size = 300
        x1, y1 = int(w/2 - box_size/2), int(h/2 - box_size/2)
        x2, y2 = x1 + box_size, y1 + box_size
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract and preprocess ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_expanded = np.expand_dims(roi_normalized, axis=0)
        
        # Predict
        preds = model.predict(roi_expanded, verbose=0)[0]
        top_idx = np.argmax(preds)
        top_conf = preds[top_idx]
        top_label = CLASSES[top_idx]
        
        # Smoothing logic
        if top_conf > CONFIDENCE_THRESHOLD:
            prediction_history.append(top_label)
            if len(prediction_history) > SMOOTHING_WINDOW:
                prediction_history.pop(0)
            
            # Check if all recent predictions match
            if len(prediction_history) == SMOOTHING_WINDOW and all(p == top_label for p in prediction_history):
                final_label = top_label
                
                # Handle special classes
                display_text = final_label
                if final_label == "space":
                    display_text = "[SPACE]"
                elif final_label == "del":
                    display_text = "[DEL]"
                elif final_label == "nothing":
                    display_text = "..."
                
                # Speak if new or cooldown passed
                current_time = time.time()
                if final_label not in ["nothing", "space", "del"] and (current_time - last_speech_time > SPEECH_COOLDOWN):
                    # Use threading to not block video loop
                    threading.Thread(target=speak, args=(engine, final_label)).start()
                    last_speech_time = current_time
            else:
                display_text = "..."
        else:
            display_text = "..."

        # Display result
        cv2.putText(frame, f"Pred: {display_text} ({top_conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("ASL Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
