# =============================================================================
# WASTE IMAGE CLASSIFICATION MODULE
# Uses CNN (ResNet50) for waste type classification (Kaggle dataset: O, R)
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image


class WasteImageClassifier:
    """
    CNN-based image classifier for waste categorization
    (Kaggle dataset with 2 classes: O and R)
    """

    def __init__(self, img_size=224, classes=None):
        self.img_size = img_size

        # For your Kaggle dataset you currently have 2 folders: 'O' and 'R'
        if classes is None:
            self.classes = ['O', 'R']
        else:
            self.classes = classes

        self.num_classes = len(self.classes)
        self.model = None

    # -------------------------------------------------------------------------
    # MODEL BUILDING
    # -------------------------------------------------------------------------
    def build_model(self):
        """Build transfer learning model using ResNet50"""

        # Load pre-trained ResNet50 (without top layer)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        # 🔴 THIS is the line you asked about:
        # Output layer = number of classes (2 for O, R)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        print("✓ Model built successfully")
        print(f"Total parameters: {self.model.count_params():,}")

        return self.model

    # -------------------------------------------------------------------------
    # DATA GENERATORS
    # -------------------------------------------------------------------------
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators for training"""

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes  # force ordering: ['O', 'R']
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes
        )

        print("Class indices:", train_generator.class_indices)
        return train_generator, val_generator

    # -------------------------------------------------------------------------
    # TRAINING / FINE-TUNING
    # -------------------------------------------------------------------------
    def train(self, train_generator, val_generator, epochs=20):
        """Train the model"""

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_waste_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def fine_tune(self, train_generator, val_generator, epochs=10):
        """Fine-tune the model by unfreezing some layers"""

        # Unfreeze last 20 layers of base model
        for layer in self.model.layers[-20:]:
            layer.trainable = True

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs
        )

        return history

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    def predict_image(self, image_path):
        """Predict waste type from a single image"""

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = self.model.predict(img_array)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx] * 100)

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
        }

    # -------------------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------------------
    def save_model(self, filepath='models/waste_classifier_final.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='models/waste_classifier_final.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")


# =============================================================================
# MAIN: TRAIN ON KAGGLE DATASET
# =============================================================================

if __name__ == "__main__":
    # 1. Create classifier (2 classes: O, R)
    classifier = WasteImageClassifier(img_size=224)

    # ⛔ IMPORTANT: CHANGE THESE TWO PATHS ACCORDING TO YOUR FOLDER
    # Example if you extracted like: C:\Users\yasha\OneDrive\Desktop\OSS\archive\DATASET\TRAIN
    # and you are running python from C:\Users\yasha\OneDrive\Desktop\OSS
    train_dir = r"archive\DATASET\TRAIN"
    val_dir   = r"archive\DATASET\TEST"
    # You can also use forward slashes:
    # train_dir = "archive/DATASET/TRAIN"
    # val_dir   = "archive/DATASET/TEST"

    # 2. Build model
    model = classifier.build_model()

    # 3. Create generators from Kaggle dataset
    train_gen, val_gen = classifier.create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32
    )

    # 4. Train
    history = classifier.train(train_gen, val_gen, epochs=20)

    # 5. Fine-tune (optional)
    history_ft = classifier.fine_tune(train_gen, val_gen, epochs=10)

    # 6. Save model
    classifier.save_model("models/waste_classifier_final.h5")
