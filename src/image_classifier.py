"""
Image classification module for NYC Rat Risk Intelligence Platform.

This module implements a CNN-based classifier to detect signs of rat
activity in images (droppings, burrows, gnaw marks, actual rats).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RatEvidenceDataset(Dataset):
    """Dataset for rat evidence images."""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform=None,
        augment: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of class labels
            transform: torchvision transforms
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Augmentation pipeline
        if augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=config.AUGMENTATION_CONFIG["horizontal_flip"]),
                A.VerticalFlip(p=config.AUGMENTATION_CONFIG["vertical_flip"]),
                A.Rotate(limit=config.AUGMENTATION_CONFIG["rotation_limit"], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.AUGMENTATION_CONFIG["brightness_limit"],
                    contrast_limit=config.AUGMENTATION_CONFIG["contrast_limit"],
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=config.AUGMENTATION_CONFIG["blur_limit"], p=0.3),
                A.RandomResizedCrop(
                    height=config.IMAGE_SIZE[0],
                    width=config.IMAGE_SIZE[1],
                    scale=(0.8, 1.0),
                    p=0.5,
                ),
            ])
        else:
            self.aug_transform = None
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # Apply augmentation
        if self.aug_transform is not None:
            augmented = self.aug_transform(image=image)
            image = augmented["image"]
            
        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


class RatEvidenceClassifier(nn.Module):
    """CNN classifier for rat evidence detection."""
    
    def __init__(
        self,
        num_classes: int = 5,
        architecture: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of output classes
            architecture: Backbone architecture
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Load pretrained backbone
        if architecture == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif architecture == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ImageClassifierTrainer:
    """Trainer for the rat evidence classifier."""
    
    def __init__(
        self,
        model: RatEvidenceClassifier,
        class_names: List[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The classifier model
            class_names: List of class names
        """
        self.model = model.to(DEVICE)
        self.class_names = class_names or config.IMAGE_CLASSES
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD,
            ),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD,
            ),
        ])
        
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
    def train(
        self,
        train_paths: List[Path],
        train_labels: List[int],
        val_paths: List[Path] = None,
        val_labels: List[int] = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        early_stopping_patience: int = None,
        freeze_backbone_epochs: int = None,
    ):
        """
        Train the classifier.
        
        Args:
            train_paths: Training image paths
            train_labels: Training labels
            val_paths: Validation image paths
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            freeze_backbone_epochs: Number of epochs to freeze backbone
        """
        logger.info("Training image classifier...")
        
        # Get parameters
        epochs = epochs or config.CLASSIFIER_TRAINING["epochs"]
        batch_size = batch_size or config.CLASSIFIER_TRAINING["batch_size"]
        learning_rate = learning_rate or config.CLASSIFIER_TRAINING["learning_rate"]
        patience = early_stopping_patience or config.CLASSIFIER_TRAINING["early_stopping_patience"]
        freeze_epochs = freeze_backbone_epochs or config.CLASSIFIER_TRAINING["freeze_backbone_epochs"]
        
        # Create datasets
        train_dataset = RatEvidenceDataset(
            train_paths, train_labels,
            transform=self.train_transform,
            augment=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        if val_paths is not None and val_labels is not None:
            val_dataset = RatEvidenceDataset(
                val_paths, val_labels,
                transform=self.val_transform,
                augment=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            val_loader = None
            
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.CLASSIFIER_TRAINING["weight_decay"],
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5,
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Freeze/unfreeze backbone
            if epoch < freeze_epochs:
                self.model.freeze_backbone()
            else:
                self.model.unfreeze_backbone()
                
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["train_acc"].append(train_acc)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100.0 * val_correct / val_total
                
                self.training_history["val_loss"].append(avg_val_loss)
                self.training_history["val_acc"].append(val_acc)
                
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            if (epoch + 1) % 5 == 0:
                val_str = f", val_loss: {avg_val_loss:.4f}, val_acc: {val_acc:.2f}%" if val_loader else ""
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.2f}%{val_str}"
                )
                
        # Load best model
        if hasattr(self, "best_state"):
            self.model.load_state_dict(self.best_state)
            
        logger.info("Training complete")
        
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted class, confidence, all class probabilities)
        """
        self.model.eval()
        
        # Transform
        image_tensor = self.val_transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Get prediction
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()
        
        # All probabilities
        all_probs = {
            self.class_names[i]: probabilities[i].item()
            for i in range(len(self.class_names))
        }
        
        return self.class_names[predicted_idx], confidence, all_probs
    
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """Predict classes for multiple images."""
        results = []
        for image in images:
            pred_class, confidence, _ = self.predict(image)
            results.append((pred_class, confidence))
        return results
    
    def evaluate(
        self,
        test_paths: List[Path],
        test_labels: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            test_paths: Test image paths
            test_labels: Test labels
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        test_dataset = RatEvidenceDataset(
            test_paths, test_labels,
            transform=self.val_transform,
            augment=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()
        
        # Per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            output_dict=True,
        )
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
        }
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "architecture": self.model.architecture,
            "num_classes": self.model.num_classes,
            "class_names": self.class_names,
            "training_history": self.training_history,
        }, path)
        logger.info(f"Saved classifier to {path}")
        
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=DEVICE)
        
        self.class_names = checkpoint["class_names"]
        self.model = RatEvidenceClassifier(
            num_classes=checkpoint["num_classes"],
            architecture=checkpoint["architecture"],
            pretrained=False,
        ).to(DEVICE)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"Loaded classifier from {path}")


def load_classifier(path: str = None) -> ImageClassifierTrainer:
    """
    Load a trained classifier.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded ImageClassifierTrainer
    """
    path = path or str(config.CLASSIFIER_MODELS_DIR / "classifier.pt")
    
    # Create dummy model
    model = RatEvidenceClassifier(
        num_classes=len(config.IMAGE_CLASSES),
        architecture=config.CLASSIFIER_ARCHITECTURE,
        pretrained=False,
    )
    
    trainer = ImageClassifierTrainer(model)
    trainer.load(path)
    
    return trainer


def classify_image(image: Image.Image, trainer: ImageClassifierTrainer = None) -> Dict:
    """
    Classify an image for rat evidence.
    
    Args:
        image: PIL Image to classify
        trainer: Loaded classifier (loads default if None)
        
    Returns:
        Dictionary with prediction results
    """
    if trainer is None:
        trainer = load_classifier()
        
    pred_class, confidence, all_probs = trainer.predict(image)
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": all_probs,
        "is_rat_evidence": pred_class != "no_evidence",
    }


if __name__ == "__main__":
    # Test classifier with synthetic data
    logger.info("Testing image classifier...")
    
    # Create a simple test
    model = RatEvidenceClassifier(
        num_classes=5,
        architecture="resnet18",
        pretrained=True,
    )
    
    trainer = ImageClassifierTrainer(model)
    
    # Test forward pass with random image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    pred_class, confidence, probs = trainer.predict(test_image)
    
    print(f"Predicted: {pred_class} (confidence: {confidence:.2%})")
    print(f"All probabilities: {probs}")
