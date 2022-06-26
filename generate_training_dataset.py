import json
import os
import shutil

file_content = open("files.json", "r")
all_files_str = file_content.read()

all_files = json.loads(all_files_str)


for key in all_files:
    files = all_files[key]

    div = round((len(files)/10) * 2)

    print(key, len(files), div)

    for i in range(0,div):
        print(i)
        source_file_name = files[i]
        if(not os.path.exists(source_file_name)):
            print("Path Not found:", source_file_name)
            continue
        target_file_name = source_file_name.replace("/training/", "/validation/")

        target_folder_split = target_file_name.split("/")
        target_folder_split.pop()
        target_folder = "/".join(target_folder_split)
        os.makedirs(target_folder,exist_ok=True)

        print(source_file_name, target_file_name, target_folder)
        shutil.move(source_file_name, target_folder)

