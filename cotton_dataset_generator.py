from PIL import Image
import psycopg2
import boto3
import json
import os
import sys
import numpy as np
import traceback

def get_aws_secret_value():
    client = boto3.client('secretsmanager', region_name="us-west-2")
    response = client.get_secret_value(
        SecretId='thulir-globals'
    )
    secret_value = json.loads(response["SecretString"])
    return secret_value

secret_value = get_aws_secret_value()

def get_connection():
    try:
        return psycopg2.connect(
            user=secret_value["postgresUserName"],
            password=secret_value["postgresPassword"],
            host=secret_value["postgresHost"],
            port=secret_value["postgresPort"]
        )
    except:
        return False

conn = get_connection()

def get_images_list():
    curr = conn.cursor()
    curr.execute("SELECT * FROM sourceimages;")
    data = curr.fetchall()

    images_list = []
    for row in data:
        s3Path = row[2]
        if ".mp4" in s3Path:
            continue

        images_list.append({
            "imageId": row[0],
            "dateTime": row[1],
            "s3Path": s3Path,
            "localFilePath": row[3],
            "imageIndex": row[4],
        })
    return images_list

def get_image_Labels(image_id):
    curr = conn.cursor()
    curr.execute(f"SELECT * FROM imagelabels where imageid='{image_id}';")
    data = curr.fetchall()

    labels = []
    for row in data:
        labels.append({
            "imageId": row[0],
            "labels": row[1]
        })

    if(len(labels) > 0):
        return labels[0]

    return {
        "imageId": "image_id",
        "labels": []
    }

images_list = get_images_list()

images_by_label = {

}

label_remap = {
    "arugampulweed": "weed",
    "damagedboll": "boll",
    "drygrassweed": "dryweed",
    "dryriceplantroot": "dryweed",
    "gressweed": "weed"
}

for image_info in images_list:
    try:
        full_path = "/Users/rajanp/datasets_local/" + image_info["s3Path"]
        extracted_dataset_path= "/Users/rajanp/extracted_datasets_local/train/"

        if(not os.path.isfile(full_path)):
            print("Not a valid path:" + full_path)
            continue

        image_labels = get_image_Labels(image_info["imageId"])
        image = Image.open(full_path)
        img_arr = np.array(image)

        for label in image_labels["labels"]:
            label_name = label["label"]
            if len(label["points"]) < 2:
                print("No Labels found for " + image_info["imageId"])
                continue

            print(label_name)
            if label_name in label_remap:
                label_name=label_remap[label_name]

            crop_path = extracted_dataset_path + label_name + "/"
            crop_file_name = crop_path + str(image_info["imageIndex"])  + "-" + str(image_info["imageId"])  + "-" + str(label["id"])  + "-" + image_info["s3Path"].replace("/", "-")
            print(crop_file_name)

            os.makedirs(crop_path, exist_ok=True)

            x_offset = label["points"][0]["x"]
            y_offset = label["points"][0]["y"]
            width = label["points"][1]["x"]
            height = label["points"][1]["y"]

            if (width < x_offset):
                tmp = x_offset
                width = x_offset
                x_offset = tmp

            if (height < y_offset):
                tmp = y_offset
                height = y_offset
                y_offset = tmp

            if (x_offset < 0):
                x_offset = 0

            if (y_offset < 0):
                y_offset = 0

            if height > image.height:
                height = image.height

            if width > image.width:
                width = image.width

            if x_offset == width:
                print("WWWWWWWWWIIIiiIIIIIIIIIIRRRRRR")
                continue

            box = (x_offset, y_offset, width, height)
            print(box, image)

            crop = image.crop(box)
            try:
                crop.save(crop_file_name, "jpeg", quality=100)
            except Exception as e:
                print(e)
                print("Error - ", image_info["imageId"])
                sys.exit(0)
                pass

            if label_name not in images_by_label:
                images_by_label[label_name] = []

            images_by_label[label_name].append(crop_file_name)
    except  Exception as ex2:
        print(ex2)
        print(traceback.format_exc())
        print("ERRRRRR:", image_info["s3Path"])
        # sys.exit(0)


f = open("files.json", "w")
f.write(json.dumps(images_by_label, indent=2))
f.close()
