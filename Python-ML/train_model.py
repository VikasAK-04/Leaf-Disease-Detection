from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
import json

# -----------------------------
# Paths and settings
# -----------------------------
train_dir = r"D:\Leaf Disease Detection\PlantVillages\dataset\train"
test_dir = r"D:\Leaf Disease Detection\PlantVillages\dataset\test"
batch_size = 32
img_size = (224, 224)
epochs = 50

print("=" * 60)
print("🌿 LEAF DISEASE DETECTION MODEL TRAINING")
print("=" * 60)

# -----------------------------
# Training data with augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Validation data WITHOUT augmentation (CRITICAL FIX)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Print dataset info
# -----------------------------
print("\n📊 DATASET INFORMATION:")
print(f"Total training samples: {train_generator.samples}")
print(f"Total validation samples: {val_generator.samples}")
print(f"Total test samples: {test_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class mapping: {train_generator.class_indices}")
print(f"Samples per class in training: {np.bincount(train_generator.classes)}")

# Save class mapping for Flask
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=4)
print("\n✅ Class indices saved to 'class_indices.json'")

# -----------------------------
# Compute class weights
# -----------------------------
y_train = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print(f"\n⚖️ Class weights (for imbalanced data): {class_weights}")

# -----------------------------
# Build Model - Two Stage Training
# -----------------------------
print("\n🏗️ BUILDING MODEL...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze all base layers initially

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# STAGE 1: Train only top layers
# -----------------------------
print("\n" + "=" * 60)
print("🎯 STAGE 1: Training classifier layers only")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_stage1 = callbacks.ModelCheckpoint(
    "stage1_best_model.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop_stage1 = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage1 = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history_stage1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint_stage1, early_stop_stage1, reduce_lr_stage1],
    class_weight=class_weights,
    verbose=1
)

# -----------------------------
# STAGE 2: Fine-tune last layers
# -----------------------------
print("\n" + "=" * 60)
print("🔥 STAGE 2: Fine-tuning MobileNetV2 layers")
print("=" * 60)

# Unfreeze last 50 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_stage2 = callbacks.ModelCheckpoint(
    "best_model.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop_stage2 = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage2 = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

history_stage2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint_stage2, early_stop_stage2, reduce_lr_stage2],
    class_weight=class_weights,
    verbose=1
)

# -----------------------------
# Evaluate on test set
# -----------------------------
print("\n" + "=" * 60)
print("📈 FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# -----------------------------
# Save final model
# -----------------------------
model.save("model.h5")
print("\n✅ Final model saved as 'model.h5'")

# -----------------------------
# Show predictions on test samples
# -----------------------------
print("\n" + "=" * 60)
print("🔍 SAMPLE PREDICTIONS (First 10 test images)")
print("=" * 60)

test_generator.reset()
predictions = model.predict(test_generator, steps=1, verbose=0)
true_classes = test_generator.classes[:batch_size]
predicted_classes = np.argmax(predictions, axis=1)

# Get class names
class_names = list(train_generator.class_indices.keys())

print("\n{:<5} {:<30} {:<30} {:<10}".format("No.", "True Label", "Predicted Label", "Confidence"))
print("-" * 80)

for i in range(min(10, len(true_classes))):
    true_label = class_names[true_classes[i]]
    pred_label = class_names[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    status = "✅" if true_classes[i] == predicted_classes[i] else "❌"
    print(f"{status} {i+1:<3} {true_label:<30} {pred_label:<30} {confidence:.2%}")

# -----------------------------
# Training summary
# -----------------------------
print("\n" + "=" * 60)
print("📝 TRAINING SUMMARY")
print("=" * 60)
  
print(f"Stage 1 Best Val Accuracy: {max(history_stage1.history['val_accuracy'])*100:.2f}%")
print(f"Stage 2 Best Val Accuracy: {max(history_stage2.history['val_accuracy'])*100:.2f}%")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print("\n✅ Training complete! Use 'model.h5' in your Flask app.")
print("=" * 60)