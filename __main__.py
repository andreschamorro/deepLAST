#!/usr/bin/env python3

"""
"""
import click
from models import training
from models import singleworker_optimizer

@click.command("train")
def train():
    return training.run()

@click.command("optimizer")
def optimizer():
    return singleworker_optimizer.run()

@click.group()
@click.pass_context
def main(ctx):
    pass

main.add_command(train)
main.add_command(optimizer)

if __name__ == "__main__":
    main()
