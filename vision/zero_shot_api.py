from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from PIL import Image
import requests
from google import genai
from dotenv import load_dotenv
import os
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import datetime

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# Define the test data directory
test_data_dir = "C:/Users/LenovoGamerNotebook/Desktop/TDK/archive/dataset1000"

# Define the available labels
labels = {
    0: "Abdominal_CT",
    1: "Breast_MRI",
    2: "CXR",
    3: "Hand_Xray",
    4: "Head_CT",
    5: "Chest_CT",
}

# Reverse mapping for label name to ID
label_name_to_id = {
    "AbdomenCT": 0,
    "BreastMRI": 1,
    "ChestCT": 5,
    "CXR": 2,  # Alternative name for ChestCT
    "Hand": 3,
    "HeadCT": 4,
    "Head_CT": 4,  # Alternative name

}


def load_true_labels():
    """Load true labels from image_labels.txt file."""
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


# Create a prompt template for Gemini
labels_list = ", ".join([f"{k}: {v}" for k, v in labels.items()])
prompt_template = f"""
Analyze this medical image and select the most suitable label from the following options if you can't recognize it choose the label you are most confident in:
{labels_list}

Please respond with only the number and label (e.g., "2: CXR") that best describes this medical image.
Look at the anatomical structures, imaging modality, and visual characteristics to make your decision.
"""


def analyze_image(image_path):
    """Analyze a single image with Gemini Pro 2."""
    try:
        # Load and process image
        image = Image.open(image_path)
        image_base64 = image_to_base64(image)

        # Make API call to Gemini Pro 2
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                {
                    "parts": [
                        {"text": prompt_template},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        )

        return response.text.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def parse_prediction(prediction_text):
    """Parse Gemini's prediction to extract label ID and name."""
    try:
        # Look for pattern like "2: CXR" or just "CXR"
        if ':' in prediction_text:
            parts = prediction_text.split(':')
            if len(parts) >= 2:
                label_id = int(parts[0].strip())
                label_name = parts[1].strip()
                return label_id, label_name

        # If no colon, try to match label names
        for label_name, label_id in label_name_to_id.items():
            if label_name.lower() in prediction_text.lower():
                return label_id, labels[label_id]

        return None, prediction_text
    except:
        return None, prediction_text


def save_accuracy_chart(class_stats, overall_accuracy, filename='accuracy_report.pdf'):
    # Prepare data for bars
    classes = []
    accuracies = []

    for class_name, stats in class_stats.items():
        if stats['total'] > 0:
            classes.append(class_name)
            class_accuracy = (stats['correct'] / stats['total']) * 100
            accuracies.append(class_accuracy)

    # Add overall accuracy bar
    classes.append('Overall')
    accuracies.append(overall_accuracy)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, accuracies, color='grey')
    plt.ylim(70, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class and Overall Accuracy')

    # Label each bar with the percentage
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f'{acc:.1f}%',
                 ha='center', color='white', fontsize=12)

    # Rotate x-labels if needed
    plt.xticks(rotation=45, ha='right')

    # Save as PDF
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Accuracy chart saved as {filename}")


def calculate_statistics(correct, total):
    """Calculate and return statistics."""
    if total == 0:
        return 0.0, 0.0

    accuracy = (correct / total) * 100
    error_rate = ((total - correct) / total) * 100

    return accuracy, error_rate


# Get all image files from test_data directory
image_files = []
if os.path.exists(test_data_dir):
    for filename in os.listdir(test_data_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_files.append(os.path.join(test_data_dir, filename))

    image_files.sort()  # Sort for consistent order
    print(f"Found {len(image_files)} images in test_data directory")
else:
    print(f"Test data directory not found: {test_data_dir}")
    exit(1)

# Load true labels
true_labels = load_true_labels()

# Analyze all images
print("\nGemini Pro 2 Analysis Results:")
print("=" * 80)
print(f"Available labels: {labels_list}")
print("=" * 80)

results = []
correct_predictions = 0
total_predictions = 0

for i, image_path in enumerate(image_files, 1):
    filename = os.path.basename(image_path)
    print(f"\n[{i}/{len(image_files)}] Analyzing: {filename}")

    # Get true label
    true_label = true_labels.get(filename, "Unknown")
    true_label_id = label_name_to_id.get(true_label, None)

    # Get prediction
    prediction_text = analyze_image(image_path)
    pred_label_id, pred_label_name = parse_prediction(prediction_text)

    # Check if prediction is correct
    is_correct = False
    if true_label_id is not None and pred_label_id is not None:
        is_correct = (true_label_id == pred_label_id)
        if is_correct:
            correct_predictions += 1
        total_predictions += 1

    # Calculate running statistics
    accuracy, error_rate = calculate_statistics(correct_predictions, total_predictions)

    # Display results
    print(f"True Label:      {true_label} (ID: {true_label_id})")
    print(f"Predicted:       {prediction_text}")
    print(f"Parsed Pred:     {pred_label_name} (ID: {pred_label_id})")
    print(f"Correct:         {'✓ YES' if is_correct else '✗ NO'}")
    print(f"Running Stats:   {correct_predictions}/{total_predictions} correct ({accuracy:.1f}% accuracy)")

    # Store results
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



# Final Summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY:")
print("=" * 80)

# Detailed results
print("\nDetailed Results:")
print("-" * 80)
for result in results:
    status = "✓" if result['is_correct'] else "✗"
    print(f"{status} {result['filename']}: {result['true_label']} → {result['pred_label_name']}")

# Overall statistics
final_accuracy, final_error_rate = calculate_statistics(correct_predictions, total_predictions)
print(f"\nOverall Statistics:")
print("-" * 80)
print(f"Total images analyzed: {len(results)}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {final_accuracy:.2f}%")
print(f"Error rate: {final_error_rate:.2f}%")

class_stats = {}
# Convert class_stats to DataFrame
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

# Per-class breakdown
print(f"\nPer-Class Breakdown:")
print("-" * 80)

for result in results:
    true_class = result['true_label']
    if true_class not in class_stats:
        class_stats[true_class] = {'total': 0, 'correct': 0}

    class_stats[true_class]['total'] += 1
    if result['is_correct']:
        class_stats[true_class]['correct'] += 1

for class_name, stats in class_stats.items():
    if stats['total'] > 0:
        class_accuracy = (stats['correct'] / stats['total']) * 100
        print(f"{class_name}: {stats['correct']}/{stats['total']} ({class_accuracy:.1f}%)")




pdf_filename = f'accuracy_report_{timestamp}.pdf'
save_accuracy_chart(class_stats, final_accuracy, filename=pdf_filename)
print(f"Accuracy chart saved as {pdf_filename}")
