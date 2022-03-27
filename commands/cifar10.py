import click

@click.group()
def cifar10():
    pass

@click.group()
def train():
    pass

@click.command()
def aaaa():
    click.echo('Train Model')



train.add_command(aaaa)

cifar10.add_command(train)