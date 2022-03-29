import click

from commands.cifar10 import cifar10
from commands.mnist import mnist
from commands.cats_vs_dogs import cats_vs_dogs

@click.group()
def cli():
    pass

cli.add_command(cifar10)
cli.add_command(mnist)
cli.add_command(cats_vs_dogs)

if __name__ == '__main__':
    cli()