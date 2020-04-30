import numpy as np
import matplotlib.pyplot as plt
import click

@click.group()
def cli():
    pass

@click.command(name="greetings")
@click.argument("name") #sam
@click.option("--age","-age", default = 20, help="how old you are") #--age 21
def greeting(name,age):
    click.echo("hello %s old %s" %(age,name))

@click.command(name="birthdays")
def birthday():
    click.echo("happy birthday")

cli.add_command(greeting)
cli.add_command(birthday)

if __name__ == "__main__":
    cli()

#python path_to_file/click_test.py greetings sam --age 21
#this can be run as an example, without the "--age 21" behind it, the default settings set it to 20.
