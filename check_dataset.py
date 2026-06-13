import os

train_path = "archive/train"
test_path = "archive/test"

print("\n===== TRAIN DATASET =====\n")

for emotion in os.listdir(train_path):
    folder_path = os.path.join(train_path, emotion)

    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        print(f"{emotion}: {count} images")

print("\n===== TEST DATASET =====\n")

for emotion in os.listdir(test_path):
    folder_path = os.path.join(test_path, emotion)

    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        print(f"{emotion}: {count} images")
        