from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_directory(
    "archive/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    "archive/test",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

# Load pretrained ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build model
model = Sequential()

model.add(base_model)
model.add(GlobalAveragePooling2D())

model.add(BatchNormalization())

model.add(Dense(
    512,
    activation="relu"
))

model.add(Dropout(0.5))

model.add(Dense(
    256,
    activation="relu"
))

model.add(Dropout(0.3))

model.add(Dense(
    7,
    activation="softmax"
))

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Show summary
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Save model
model.save("resnet_emotion.keras")

print("ResNet model saved successfully!")