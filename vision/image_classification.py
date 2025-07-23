import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.fx.experimental.graph_gradual_typechecker import transpose_inference_rule
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import platform
import psutil
import sys
from datetime import datetime
from tqdm import tqdm


class ImageClassifier:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Initialize configuration variables
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']
        self.augmentation_config = self.config['augmentation']
        self.device = self.config['device']
        self.seed = self.config['seed']
        self.num_workers = self.config['num_workers']
        self.augmented = self.config['data']['augmented']

        self.initialize_backend()
        self.prepare_dataset()
        self.log_training_info()


    def initialize_backend(self) -> None:
        """
        Initializes the backend system.

        This method sets up the necessary components or configurations
        required for the backend system to function.

        Sets random seeds for PyTorch and NumPy to ensure reproducibility.
        Initializes the model based on the configuration.

        :return: None
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize model based on configuration
        architecture = self.model_config['architecture']
        pretrained = self.model_config['pretrained']
        num_classes = self.model_config['num_classes']

        # Select model architecture
        if architecture.lower() == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            # Modify the final layer for CIFAR-10 (10 classes)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif architecture.lower() == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            # Modify the final layer for CIFAR-10 (10 classes)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")

        # Move model to the specified device
        self.device = torch.device(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
        self.model = self.model.to(self.device)

    def augmentation_transform(self, image_size, mean, std):
        """

        :param image_size:
        :param mean:
        :param std:
        :return:
        """

        transform = transforms.Compose([
            transforms.Resize(image_size),
            # Add random crop if enabled in config
            transforms.RandomCrop(image_size, padding=4) if self.augmentation_config[
                'random_crop'] else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(p=0.5 if self.augmentation_config['horizontal_flip'] else 0),
            transforms.RandomVerticalFlip(p=0.5 if self.augmentation_config['vertical_flip'] else 0),
            transforms.RandomRotation(
                self.augmentation_config['random_rotation'] if self.augmentation_config['random_rotation'] else 0),
            transforms.ColorJitter(
                brightness=self.augmentation_config['color_jitter']['brightness'],
                contrast=self.augmentation_config['color_jitter']['contrast'],
                saturation=self.augmentation_config['color_jitter']['saturation'],
                hue=self.augmentation_config['color_jitter']['hue']
            ) if 'color_jitter' in self.augmentation_config else transforms.ColorJitter(0, 0, 0, 0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return transform

    def basic_transform(self, image_size, mean, std):
        """

        :param image_size:
        :param mean:
        :param std:
        :return:
        """
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return transform

    def load_data(self, data_dir, train_transform, test_transform):

        self.train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        self.test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

    def prepare_dataset(self) -> None:
        """
        Prepares the CIFAR-10 dataset for processing. This method downloads the dataset,
        applies transformations, and creates dataloaders for training and testing.

        The dataset is downloaded to '../data/cifar_10' directory.
        Transformations are applied based on the configuration in config.json.

        :return: None
        :rtype: None
        """
        # Create directory for dataset
        data_dir = os.path.join('..', 'data', 'cifar_10')
        os.makedirs(data_dir, exist_ok=True)

        # Get image size, mean, and std from config
        image_size = tuple(self.data_config['image_size'])
        mean = tuple(self.data_config['mean'])
        std = tuple(self.data_config['std'])

        # Define transformations for training data
        if self.augmented:
            train_transform = self.augmentation_transform(image_size, mean, std)
        else:
            train_transform = self.basic_transform(image_size, mean, std)

        # Define transformations for testing data (no augmentation)
        test_transform = self.basic_transform(image_size, mean, std)

        # Download and load the CIFAR-10 dataset
        self.load_data(data_dir, train_transform, test_transform)

    def log_training_info(self) -> None:
        """
        Logs training information such as progress or metrics to be monitored during
        the training process. This method is intended to be used to ensure proper 
        tracking and analysis of the training phase data.

        :return: None
        """
        # Configure logger
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # Set up logging format
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)


        # Log system information
        logger.info("\n" + "=" * 30 + " SYSTEM INFORMATION " + "=" * 30)

        # OS Information
        logger.info(f"OS: {platform.system()} {platform.release()} ({platform.platform()})")
        logger.info(f"Python Version: {platform.python_version()}")

        # CPU Information
        logger.info(f"CPU: {platform.processor()}")
        logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count(logical=True)} (Logical)")
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB (Total), {psutil.virtual_memory().available / (1024**3):.2f} GB (Available)")

        # GPU Information
        if torch.cuda.is_available():
            logger.info(f"CUDA Available: Yes")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        else:
            logger.info(f"CUDA Available: No")

        # Threads Information
        logger.info(f"Number of Workers: {self.num_workers}")

        # Log hyperparameters
        logger.info("\n" + "=" * 30 + " HYPERPARAMETERS " + "=" * 30)

        # Model Configuration
        logger.info("\nModel Configuration:")
        for key, value in self.model_config.items():
            logger.info(f"  {key}: {value}")

        # Training Configuration
        logger.info("\nTraining Configuration:")
        for key, value in self.training_config.items():
            logger.info(f"  {key}: {value}")

        # Data Configuration
        logger.info("\nData Configuration:")
        for key, value in self.data_config.items():
            logger.info(f"  {key}: {value}")

        # Augmentation Configuration
        logger.info("\nAugmentation Configuration:")
        for key, value in self.augmentation_config.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

        # Other Configuration
        logger.info("\nOther Configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")
        logger.info(f"  Number of Workers: {self.num_workers}")

        logger.info("\n" + "=" * 80)
        logger.info(f"{'TRAINING CONFIGURATION COMPLETE':^80}")
        logger.info("=" * 80)

    def train(self) -> None:
        """
        Trains the model using the training dataset.

        This method implements the training loop for the specified number of epochs.
        It tracks and logs training loss, training accuracy, test loss, and test accuracy
        after each epoch.

        :return: None
        """
        # Get logger
        logger = logging.getLogger(__name__)

        # Set model to training mode
        self.model.train()

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Configure optimizer based on config
        optimizer_name = self.training_config['optimizer'].lower()
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config['weight_decay']

        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.training_config['momentum']
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Configure learning rate scheduler if specified
        scheduler = None
        if 'scheduler' in self.training_config and self.training_config['scheduler'] == 'step':
            step_size = self.training_config['step_size']
            gamma = self.training_config['gamma']
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Get number of epochs
        num_epochs = self.training_config['num_epochs']

        # Log training start
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info("=" * 80)

        # Training loop
        for epoch in range(num_epochs):
            # Initialize metrics for this epoch
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over batches with tqdm
            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, colour="white"):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # Calculate average training loss and accuracy for this epoch
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total

            # Evaluate on test set
            test_metrics = self.evaluate()
            test_loss = test_metrics['test_loss']
            test_accuracy = test_metrics['test_accuracy']

            # Log metrics for this epoch
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Train Accuracy: {train_accuracy:.2f}%, "
                        f"Test Loss: {test_loss:.4f}, "
                        f"Test Accuracy: {test_accuracy:.2f}%")

            # Update learning rate if scheduler is used
            if scheduler:
                scheduler.step()
                logger.info(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

            # Set model back to training mode for next epoch
            self.model.train()

        # Log training completion
        logger.info("=" * 80)
        logger.info("Training completed")
        logger.info("=" * 80)

    def evaluate(self) -> dict:
        """
        Evaluates the model on the test dataset.

        This method calculates the loss and accuracy of the model on the test dataset.
        It runs the model in evaluation mode to disable dropout and batch normalization.
        It also calculates and logs additional metrics: classification report by class,
        confusion matrix, F1 score, and recall.

        :return: Dictionary containing test metrics (loss, accuracy, f1_score, recall)
        :rtype: dict
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize metrics
        test_loss = 0.0
        correct = 0
        total = 0

        # Lists to store all predictions and targets for detailed metrics
        all_predictions = []
        all_targets = []

        # Get logger
        logger = logging.getLogger(__name__)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Disable gradient calculation for evaluation
        with torch.no_grad():
            # Iterate over test data with tqdm
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating", leave=False, colour="white"):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store predictions and targets for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate average loss and accuracy
        avg_test_loss = test_loss / len(self.test_loader)
        test_accuracy = 100.0 * correct / total

        # Convert lists to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)



        # Log test metrics
        logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"F1 Score (macro): {f1:.4f}, Recall (macro): {recall:.4f}")

        # Log classification report
        logger.info("\nClassification Report by Class:")
        logger.info(f"\n{class_report}")

        # Log confusion matrix with better formatting
        logger.info("\nConfusion Matrix:")
        # Add header with class indices
        header = "    " + " ".join(f"{i:4d}" for i in range(len(self.test_dataset.classes)))
        logger.info(header)
        # Log each row with class label
        for i, (row, class_name) in enumerate(zip(conf_matrix, self.test_dataset.classes)):
            formatted_row = " ".join(f"{cell:4d}" for cell in row)
            logger.info(f"{i:2d} {class_name[:5]:5s} {formatted_row}")

        # Return metrics
        return {
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'recall': recall,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }

    def test_performance(self):
        pass


if __name__ == "__main__":
    try:
        image_classifier = ImageClassifier()
        image_classifier.train()
        image_classifier.evaluate()
    except FileNotFoundError:
        print("Error: config.json file not found in the specified path")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json")
