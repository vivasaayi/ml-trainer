import json
import os
import shutil
import random
import sys

base_path = "/Users/rajanp/extracted_datasets_local-processed/train"
dirs = os.listdir(base_path)
print(dirs)

all_files = {}

for dir in dirs:
    class_dir = os.path.join(base_path, dir)
    if not os.path.isdir(class_dir):
        continue

    all_files[dir] = []

    os.makedirs(class_dir, exist_ok=True)

    for file in os.listdir(class_dir):
        full_path = os.path.join(class_dir, file)
        all_files[dir].append((full_path))

print(json.dumps(all_files, indent=2))

for key in all_files:
    files = all_files[key]

    div = round((len(files)/10) * 2)

    print(key, len(files), div)

    random_numbers = []

    for i in range(0, div):
        random_no = random.randrange(0, len(files))
        random_numbers.append(random_no)
        print(random_no)

    for i in random_numbers:
        source_file_name = files[i]
        if(not os.path.exists(source_file_name)):
            print("Path Not found:", source_file_name)
            continue
        target_file_name = source_file_name.replace("/train/", "/validation/")

        target_folder_split = target_file_name.split("/")
        target_folder_split.pop()
        target_folder = "/".join(target_folder_split)
        os.makedirs(target_folder,exist_ok=True)

        print(source_file_name, target_file_name, target_folder)
        shutil.move(source_file_name, target_folder)


