from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from PIL import Image
import requests
from google import genai
from dotenv import load_dotenv
import os
import base64
import io
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import datetime

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- REPLACE WITH YOUR ACTUAL LABEL MAPPINGS ---
labels = {
    0: "Abdominal_CT",
    1: "Breast_MRI",
    2: "CXR",
    3: "Hand_Xray",
    4: "Head_CT",
    5: "Chest_CT",
}

label_name_to_id = {
    "AbdomenCT": 0,
    "BreastMRI": 1,
    "ChestCT": 5,
    "CXR": 2,
    "Hand": 3,
    "HeadCT": 4,
    "Head_CT": 4,
}

# --- CORE HELPER FUNCTIONS ---
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_example_images(examples_dir="C:/Users/LenovoGamerNotebook/Desktop/TDK/archive/one_shot_examples"):
    example_images = {}
    for label_name in label_name_to_id.keys():
        # Try both .jpg and .jpeg extensions
        example_path = os.path.join(examples_dir, f"{label_name}.jpg")
        if not os.path.exists(example_path):
            example_path = os.path.join(examples_dir, f"{label_name}.jpeg")
        if os.path.exists(example_path):
            with Image.open(example_path) as img:
                example_images[label_name] = image_to_base64(img)
        else:
            print(f"Warning: Example image for {label_name} not found in {examples_dir}")
    return example_images

def build_oneshot_prompt(query_image_base64, example_images, labels, label_name_to_id):
    instruction = (
        "Analyze this medical image and select the most suitable label from the following options.\n"
        "If you are unsure, choose the label you are most confident in.\n"
        "Respond with only the number and label (e.g., '2: CXR').\n\n"
        "Here are examples of each class:\n"
    )
    for label_name, img_base64 in example_images.items():
        label_id = label_name_to_id[label_name]
        label_str = labels[label_id]
        instruction += f"- {label_str}: [EXAMPLE_IMAGE]\n"
    instruction += "\nOptions: " + ", ".join([f"{k}: {v}" for k, v in labels.items()]) + "\n\n"
    instruction += "Query image to classify: [QUERY_IMAGE]\n"
    return instruction

def analyze_image_one_shot(image_path, example_images, labels, label_name_to_id):
    try:
        with Image.open(image_path) as img:
            query_base64 = image_to_base64(img)
    except Exception as e:
        return f"Error opening query image: {str(e)}"

    # Build the multipart message: alternating text and images
    parts = [
        {"text": "Analyze this medical image and select the most suitable label from the following options.\n"
                 "If you are unsure, choose the label you are most confident in.\n"
                 "Respond with only the number and label (e.g., '2: CXR').\n\n"
                 "Here are examples of each class:\n"}
    ]

    # Add one example image and label per class
    for label_name, img_base64 in example_images.items():
        class_id = label_name_to_id[label_name]
        class_label = labels[class_id]
        parts.append({"text": f"- {class_label}:\n"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_base64}})
        parts.append({"text": "\n"})

    parts.append({"text": "\nOptions: " + ", ".join([f"{k}: {v}" for k, v in labels.items()]) + "\n\n"})
    parts.append({"text": "Query image to classify:\n"})
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": query_base64}})

    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[{"parts": parts}]
        )
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def parse_prediction(prediction_text):
    try:
        if ':' in prediction_text:
            parts = prediction_text.split(':')
            if len(parts) >= 2:
                label_id = int(parts[0].strip())
                label_name = parts[1].strip()
                return label_id, label_name
        for label_name, label_id in label_name_to_id.items():
            if label_name.lower() in prediction_text.lower():
                return label_id, labels[label_id]
        return None, prediction_text
    except:
        return None, prediction_text

def calculate_statistics(correct, total):
    if total == 0:
        return 0.0, 0.0
    accuracy = (correct / total) * 100
    error_rate = ((total - correct) / total) * 100
    return accuracy, error_rate

def save_accuracy_chart(class_stats, overall_accuracy, filename='accuracy_report.pdf'):
    classes = []
    accuracies = []
    for class_name, stats in class_stats.items():
        if stats['total'] > 0:
            classes.append(class_name)
            class_accuracy = (stats['correct'] / stats['total']) * 100
            accuracies.append(class_accuracy)
    classes.append('Overall')
    accuracies.append(overall_accuracy)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, accuracies, color='green')
    plt.ylim(70, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class and Overall Accuracy')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f'{acc:.1f}%',
                 ha='center', color='white', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Accuracy chart saved as {filename}")

# --- MAIN LOGIC ---

test_data_dir = "C:/Users/LenovoGamerNotebook/Desktop/TDK/archive/dataset1000"

def load_true_labels():
    labels_file = os.path.join(test_data_dir, 'image_labels_1000.txt')
    true_labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label_name = parts[1]
                        true_labels[image_name] = label_name
        print(f"Loaded {len(true_labels)} true labels from {labels_file}")
    else:
        print(f"Warning: Labels file not found at {labels_file}")
    return true_labels

# --- PREPARE EXAMPLE IMAGES ---
example_images = load_example_images()
if not example_images:
    print("ERROR: No example images found. Place one .jpg or .jpeg per class in the examples directory.")
    exit(1)

# --- COLLECT TEST IMAGES ---
image_files = []
if os.path.exists(test_data_dir):
    for filename in os.listdir(test_data_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_files.append(os.path.join(test_data_dir, filename))
    image_files.sort()
    print(f"Found {len(image_files)} images in test_data directory")
else:
    print(f"Test data directory not found: {test_data_dir}")
    exit(1)

true_labels = load_true_labels()

# --- MAIN LOOP ---
results = []
correct_predictions = 0
total_predictions = 0
class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

for i, image_path in enumerate(image_files, 1):
    filename = os.path.basename(image_path)
    print(f"\n[{i}/{len(image_files)}] Analyzing: {filename}")

    true_label = true_labels.get(filename, "Unknown")
    true_label_id = label_name_to_id.get(true_label, None)

    prediction_text = analyze_image_one_shot(image_path, example_images, labels, label_name_to_id)
    pred_label_id, pred_label_name = parse_prediction(prediction_text)

    is_correct = False
    if true_label_id is not None and pred_label_id is not None:
        is_correct = (true_label_id == pred_label_id)
        if is_correct:
            correct_predictions += 1
            class_stats[true_label]['correct'] += 1
        class_stats[true_label]['total'] += 1
        total_predictions += 1

    accuracy, error_rate = calculate_statistics(correct_predictions, total_predictions)

    print(f"True Label:      {true_label} (ID: {true_label_id})")
    print(f"Predicted:       {prediction_text}")
    print(f"Parsed Pred:     {pred_label_name} (ID: {pred_label_id})")
    print(f"Correct:         {'✓ YES' if is_correct else '✗ NO'}")
    print(f"Running Stats:   {correct_predictions}/{total_predictions} correct ({accuracy:.1f}% accuracy)")

    results.append({
        'filename': filename,
        'path': image_path,
        'true_label': true_label,
        'true_label_id': true_label_id,
        'prediction_text': prediction_text,
        'pred_label_name': pred_label_name,
        'pred_label_id': pred_label_id,
        'is_correct': is_correct
    })

# --- FINAL REPORT ---
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY:")
print("=" * 80)
print("\nDetailed Results:")
print("-" * 80)
for result in results:
    status = "✓" if result['is_correct'] else "✗"
    print(f"{status} {result['filename']}: {result['true_label']} → {result['pred_label_name']}")

final_accuracy, final_error_rate = calculate_statistics(correct_predictions, total_predictions)
print(f"\nOverall Statistics:")
print("-" * 80)
print(f"Total images analyzed: {len(results)}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {final_accuracy:.2f}%")
print(f"Error rate: {final_error_rate:.2f}%")

stats_list = []
for class_name, stats in class_stats.items():
    if stats['total'] > 0:
        accuracy_pct = (stats['correct'] / stats['total']) * 100
        stats_list.append({
            'Class': class_name,
            'Correct': stats['correct'],
            'Total': stats['total'],
            'Accuracy (%)': accuracy_pct
        })

# Create DataFrame
stats_df = pd.DataFrame(stats_list)

# Add overall stats
overall_stats = pd.DataFrame([{
    'Class': 'Overall',
    'Correct': correct_predictions,
    'Total': total_predictions,
    'Accuracy (%)': final_accuracy
}])
final_stats_df = pd.concat([stats_df, overall_stats], ignore_index=True)

# Generate timestamp for filename
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Save to CSV
csv_filename = f'accuracy_stats_{timestamp}.csv'
final_stats_df.to_csv(csv_filename, index=False)
print(f"Accuracy statistics saved to {csv_filename}")

print(f"\nPer-Class Breakdown:")
print("-" * 80)
for class_name, stats in class_stats.items():
    if stats['total'] > 0:
        class_accuracy = (stats['correct'] / stats['total']) * 100
        print(f"{class_name}: {stats['correct']}/{stats['total']} ({class_accuracy:.1f}%)")

# Convert class_stats to DataFrame


pdf_filename = f'accuracy_report_{timestamp}.pdf'
save_accuracy_chart(class_stats, final_accuracy, filename=pdf_filename)
print(f"Accuracy chart saved as {pdf_filename}")