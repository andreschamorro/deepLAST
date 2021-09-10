#!/usr/bin/env python3

"""
"""
import click
from glob import glob
from models import training
from models import singleworker_optimizer
from data_loader import processingfasta

@click.command("train")
def train():
    return training.run()

@click.command("optimizer")
def optimizer():
    return singleworker_optimizer.run()

@click.command("fastq2tfrecord")
@click.argument('fafiles', type=str, nargs=-1)
@click.option('-c', '--classes', type=str, required=True, help='classes int csv of fastafiles')
@click.option('-m', '--d2vmodel', type=str, required=True, help='dna to vector model file')
@click.option('-k', '--kmersize', type=int, default=8, help='kmer size')
@click.option('-i', '--ignore', type=str, help='drop sequence in files ids')
@click.option('-d', '--outdir', type=str, default='./tfdata', help='output dir')
def fastq2tfrecord(fafiles, classes, d2vmodel, kmersize, ignore, outdir):
    """Format fastas file to train, validation and test TFRecords"""
    fafilenames = []
    for filename in fafiles:
        expanded = list(glob(filename))
        if len(expanded) == 0 and '*' not in filename:
            raise(click.BadParameter(
                "file '{}' not found".format(filename)))
        fafilenames.extend(expanded)
    processingfasta.run(fafilenames, classes.split(','), outdir, d2vmodel, kmersize, ignore)

@click.group()
@click.pass_context
def main(ctx):
    pass

main.add_command(train)
main.add_command(optimizer)
main.add_command(fastq2tfrecord)

if __name__ == "__main__":
    main()
