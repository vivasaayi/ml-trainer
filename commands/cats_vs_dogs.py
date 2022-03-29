import click

from cats_vs_dogs.resnet50.train import do_train

@click.group()
def cats_vs_dogs():
    pass

@click.group()
def train():
    print('Training Cats VS Dogs Dataset')

    pass

@click.command()
def resnet50():
    print('Using ResNet50')
    do_train()


train.add_command(resnet50)

cats_vs_dogs.add_command(train)