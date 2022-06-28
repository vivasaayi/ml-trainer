def list_dir(name_):
  import os
  for dir_path, dirnames, filenames in os.walk(name_):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dir_path}')

list_dir("/Users/rajanp/extracted_datasets_local/")