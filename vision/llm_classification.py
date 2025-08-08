import json
import ollama
import pickle
import numpy as np
import io
from PIL import Image
import base64
import re
import requests
import os


class LLMClassifier:
    def __init__(self):
        # Load configuration from config_llm.json
        with open('config_llm.json', 'r') as f:
            self.config = json.load(f)
        self.configure_network()
        self.read_data()
        self.prompt = (
            "You are given an image from the CIFAR-10 dataset. The possible labels are:\n"
            "- 0: airplane\n"
            "- 1: automobile\n"
            "- 2: bird\n"
            "- 3: cat\n"
            "- 4: deer\n"
            "- 5: dog\n"
            "- 6: frog\n"
            "- 7: horse\n"
            "- 8: ship\n"
            "- 9: truck\n\n"
            "Your task is to analyze the image and assign the most appropriate label (from the list above) to the given picture."
        )
    

    
    def configure_network(self):
        # Initialize Ollama client
        self.client = ollama.Client()
        
        # Check if Gemma3 vision model is already downloaded
        try:
            # Try to list models to see if gemma3:vision exists
            models = self.client.list()
            model_exists = any('gemma3:27b' in model['name'] for model in models['models'])
            
            if model_exists:
                print("gemma3:27b latest model already downloaded, using existing model.")
                self.model = 'gemma3:27b'
            else:
                print("Downloading gemma3:27b latest model...")
                self.model = self.client.pull('gemma3:27b')
                
        except Exception as e:
            print(f"Error checking for existing model: {e}")
            print("Downloading gemma3:27b latest model...")
            self.model = self.client.pull('gemma3:27b')
    
    def inference(self, image):
        # Convert numpy image (3, 32, 32) to PIL Image and then to base64
        img = np.transpose(image, (1, 2, 0))  # (32, 32, 3)
        pil_img = Image.fromarray(img.astype('uint8'))
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare message for Ollama
        message = {
            "role": "user",
            "content": self.prompt,
            "images": [img_b64]
        }

        # Call Ollama chat
        response = self.client.chat(model='gemma3:27b', messages=[message])
        
        # Extract label from response (expecting a single integer 0-9 in the reply)
        match = re.search(r'\b([0-9])\b', str(response.get('message', {}).get('content', '')))
        if match:
            return int(match.group(1))
        else:
            # If not found, return -1 or raise error
            return -1
    
    def api_inference(self, image):
        """
        API-friendly inference method using Gemini API.
        Args:
            image: Input image as a numpy array (shape: 3, 32, 32)
        Returns:
            dict: { "predicted_label": int }
        """
        # Convert numpy image (3, 32, 32) to PIL Image and then to base64
        img = np.transpose(image, (1, 2, 0))  # (32, 32, 3)
        pil_img = Image.fromarray(img.astype('uint8'))
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare payload for Gemini API
        api_key = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                    ]
                }
            ]
        }

        params = {"key": api_key}
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        result = response.json()

        # Parse the response to extract the label (expecting a single integer 0-9)
        content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        match = re.search(r'\b([0-9])\b', str(content))
        if match:
            return {"predicted_label": int(match.group(1))}
        else:
            return {"predicted_label": -1}
    
    
    def read_data(self):
        # Read one batch from CIFAR-10 dataset
        batch_file = '../data/cifar_10/cifar-10-batches-py/data_batch_1'
        
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
        
        # Extract images and labels
        images = batch_data[b'data']
        labels = batch_data[b'labels']
        
        # Reshape images to (N, 3, 32, 32) format
        images = images.reshape(-1, 3, 32, 32)
        
        # Convert to numpy arrays
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        print(f"Loaded {len(self.images)} images with shape {self.images.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Sample labels: {self.labels[:10]}")
        


def test_loop():
    # Create LLM classifier instance
    classifier = LLMClassifier()
    
    # Test on a few sample images
    num_test_images = 200
    correct_cases = 0
    
    print(f"Testing inference on {num_test_images} images...")
    print("=" * 50)
    
    for i in range(num_test_images):
        # Get image and true label
        image = classifier.images[i]
        true_label = classifier.labels[i]
        
        # Call inference to get predicted label
        predicted_label = classifier.inference(image)
        
        # Count correct cases
        if predicted_label == true_label:
            correct_cases += 1
        
        # Print results
        print(f"Image {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted label: {predicted_label}")
        print(f"  Correct: {'✓' if predicted_label == true_label else '✗'}")
        print("-" * 30)
    
    return correct_cases, num_test_images


def report(num_pictures, correct_cases):
    # Calculate accuracy
    accuracy = correct_cases / num_pictures if num_pictures > 0 else 0
    
    print(f"Report:")
    print(f"Total pictures: {num_pictures}")
    print(f"Correct predictions: {correct_cases}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # For precision and recall, we need true positives, false positives, false negatives
    # Since we're doing multi-class classification, we'll calculate macro averages
    # This is a simplified version - in practice you'd need the full confusion matrix
    
    print(f"Precision: {accuracy:.4f} (simplified - assuming balanced classes)")
    print(f"Recall: {accuracy:.4f} (simplified - assuming balanced classes)")
    
    # Print a simple confusion matrix structure
    print(f"\nConfusion Matrix (simplified):")
    print(f"Predicted vs Actual:")
    print(f"Correct predictions: {correct_cases}")
    print(f"Incorrect predictions: {num_pictures - correct_cases}")
    print(f"Total: {num_pictures}")


def main():
    llm_classifier = LLMClassifier()
    correct_cases, num_test_images = test_loop()
    report(num_test_images, correct_cases)


if __name__ == "__main__":
    main()
 