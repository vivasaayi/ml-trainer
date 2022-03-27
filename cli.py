import click

from commands.cifar10 import cifar10
from commands.mnist import mnist

@click.group()
def cli():
    pass

cli.add_command(cifar10)
cli.add_command(mnist)

if __name__ == '__main__':
    cli()