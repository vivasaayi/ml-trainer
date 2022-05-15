import click

from commands.cifar10 import cifar10
from commands.mnist import mnist
from commands.rice import rice
from commands.cotton import cotton

@click.group()
def cli():
    pass

cli.add_command(cifar10)
cli.add_command(mnist)
cli.add_command(rice)
cli.add_command(cotton)

if __name__ == '__main__':
    cli()