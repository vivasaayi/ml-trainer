https://link.springer.com/chapter/10.1007/978-981-19-2095-0_45

python cli.py mltrainer train-torch-net  --model-name=local.cnn --data-loader-name RICE_HEALTH --dataset-path=/users/rajanp/Downloads/rice/RiceDiseaseDataset

 python cli.py mltrainer train-torch-net --prebuilt-dataset-name=MNIST

python cli.py mltrainer train-torch-net  --model-name=local.cnn --data-loader-name COTTON_DISEASE --dataset-path=/users/rajanp/Downloads/cotton-disease


git clone https://github.com/vivasaayi/ml-trainer.git
cd ml-trainer
pip install -r requirements.txt

cp drive/MyDrive/datasets/cotton-disease-processed.zip sample_data/
cd sample_data/
unzip cotton-disease-processed.zip

cd /content/ml-trainer
python cli.py mltrainer train-torch-net  --model-name=local.cnn --data-loader-name COTTON_DISEASE --dataset-path=/content/sample_data/cotton-disease-processed



python cli.py mltrainer train-torch-net \ 
    --model-name=inception \
    --data-loader-name COTTON_PLANTS \ 
    --dataset-path=/Users/rajanp/extracted_datasets_local


# Cotton Plants

```shell
python cli.py cottonplants preprocess prepare-dataset

python cli.py mltrainer train-torch-net  --model-name=local.cnn --data-loader-name COTTON_PLANTS --dataset-path=/Users/rajanp/extracted_datasets_local-processed

```

```shell
git clone https://github.com/vivasaayi/ml-trainer.git
cd ml-trainer
pip install -r requirements.txt

cp drive/MyDrive/datasets/cotton-disease-processed.zip sample_data/
cd sample_data/
unzip cotton-disease-processed.zip
```
