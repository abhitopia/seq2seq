#!/bin/bash

wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip
mkdir binary
mkdir finegrained
mkdir binary/train
mkdir binary/test
mkdir binary/dev
mkdir finegrained/train
mkdir finegrained/test
mkdir finegrained/dev

rm trainDevTestTrees_PTB.zip
python data.py
rm -r trees

