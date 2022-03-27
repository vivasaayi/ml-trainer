import click

from mnist.simple_convnet.simple_convnet import SimpleConvnet

@click.group()
def mnist():
    pass

@click.group()
def train():
    print('Training MNIST Dataset')

    pass

@click.command()
def simple_convnet():
    print('Using Simple ConvNet')
    trainer = SimpleConvnet()
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


train.add_command(simple_convnet)

mnist.add_command(train)