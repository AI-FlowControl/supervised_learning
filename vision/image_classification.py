import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

class Classifier():
    def __init__(self, config_path=None):
        """
        Initialize the Classifier with configuration from a JSON file.

        Args:
            config_path (str, optional): Path to the configuration JSON file.
                If None, it will try to find the config file in the current directory
                or in the vision subdirectory.
        """
        # Determine the config path
        if config_path is None:
            # Try different possible locations
            possible_paths = ['config.json', 'vision/config.json', '../vision/config.json']
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            if config_path is None:
                raise FileNotFoundError("Config file not found in any of the expected locations")
        # Load configuration from JSON file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Set object-level variables from config
        # Model parameters
        self.model_architecture = self.config['model']['architecture']
        self.pretrained = self.config['model']['pretrained']
        self.num_classes = self.config['model']['num_classes']

        # Training parameters
        self.batch_size = self.config['training']['batch_size']
        self.learning_rate = self.config['training']['learning_rate']
        self.num_epochs = self.config['training']['num_epochs']
        self.optimizer = self.config['training']['optimizer']
        self.weight_decay = self.config['training']['weight_decay']
        self.momentum = self.config['training']['momentum']
        self.scheduler = self.config['training']['scheduler']
        self.step_size = self.config['training']['step_size']
        self.gamma = self.config['training']['gamma']

        # Data parameters
        self.image_size = self.config['data']['image_size']
        self.mean = self.config['data']['mean']
        self.std = self.config['data']['std']
        self.train_dir = self.config['data']['train_dir']
        self.val_dir = self.config['data']['val_dir']
        self.test_dir = self.config['data']['test_dir']

        # Augmentation parameters
        self.horizontal_flip = self.config['augmentation']['horizontal_flip']
        self.vertical_flip = self.config['augmentation']['vertical_flip']
        self.random_crop = self.config['augmentation']['random_crop']
        self.random_rotation = self.config['augmentation']['random_rotation']
        self.color_jitter = self.config['augmentation']['color_jitter']

        # Other parameters
        self.device = self.config['device']
        self.seed = self.config['seed']
        self.num_workers = self.config['num_workers']

        self.prepare_data()

    def prepare_data(self):
        """
        Prepare data for training, validation, and testing.
        - Read data from specified directories
        - Apply transformations based on configuration
        - Create data loaders
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Define transformations for training (with augmentation)
        train_transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        # Add augmentation transforms based on config
        if self.random_crop:
            train_transform_list.insert(1, transforms.RandomCrop(self.image_size[0], padding=4))
        if self.horizontal_flip:
            train_transform_list.insert(1, transforms.RandomHorizontalFlip())
        if self.vertical_flip:
            train_transform_list.insert(1, transforms.RandomVerticalFlip())
        if self.random_rotation > 0:
            train_transform_list.insert(1, transforms.RandomRotation(self.random_rotation))
        if self.color_jitter:
            jitter = self.config['augmentation']['color_jitter']
            train_transform_list.insert(1, transforms.ColorJitter(
                brightness=jitter.get('brightness', 0),
                contrast=jitter.get('contrast', 0),
                saturation=jitter.get('saturation', 0),
                hue=jitter.get('hue', 0)
            ))

        train_transform = transforms.Compose(train_transform_list)

        # Define transformations for validation and testing (no augmentation)
        val_test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Load datasets
        try:
            train_dataset = datasets.ImageFolder(self.train_dir, transform=train_transform)
            val_dataset = datasets.ImageFolder(self.val_dir, transform=val_test_transform)
            test_dataset = datasets.ImageFolder(self.test_dir, transform=val_test_transform)

            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

            # Store class names
            self.class_names = train_dataset.classes
            print(f"Data loaded successfully. Number of classes: {len(self.class_names)}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def train(self):
        """
        Train the model using the configuration parameters.
        - Initialize model based on architecture
        - Set up loss function, optimizer, and scheduler
        - Train for specified number of epochs
        - Evaluate on validation and test sets
        - Report metrics including accuracy, confusion matrix, recall, and F1 score
        """
        # Set device
        device = torch.device(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
        print(f"Using device: {device}")

        # Initialize model
        print(f"Initializing {self.model_architecture} model...")
        if self.model_architecture == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_architecture == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_architecture == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_architecture == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        model = model.to(device)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, 
                                 momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Define scheduler
        if self.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        elif self.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        else:
            scheduler = None

        # Training loop
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print('-' * 10)

            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in self.train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)

            print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0

            # Iterate over validation data
            for inputs, labels in self.val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

            val_epoch_loss = val_running_loss / len(self.val_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(self.val_loader.dataset)

            print(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

            # Save the best model
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_val_acc:.4f}")

        print(f"Best validation accuracy: {best_val_acc:.4f}")

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Evaluate on validation set
        self.evaluate(model, device, self.val_loader, "Validation")

        # Evaluate on test set
        self.evaluate(model, device, self.test_loader, "Test")

        return model

    def evaluate(self, model, device, data_loader, dataset_name):
        """
        Evaluate the model on the given dataset and report metrics.

        Args:
            model: The trained model
            device: The device to run evaluation on
            data_loader: DataLoader for the dataset to evaluate
            dataset_name: Name of the dataset (for reporting)
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        # Calculate per-class metrics
        class_report = classification_report(all_labels, all_preds, 
                                            target_names=self.class_names, 
                                            output_dict=True)

        # Print results
        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        for cls in self.class_names:
            cls_metrics = class_report[cls]
            print(f"{cls}: Precision={cls_metrics['precision']:.4f}, "
                  f"Recall={cls_metrics['recall']:.4f}, "
                  f"F1-score={cls_metrics['f1-score']:.4f}")

        # Return metrics for further use if needed
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report
        }

if __name__ == '__main__':
    classifier = Classifier()
    classifier.train()
