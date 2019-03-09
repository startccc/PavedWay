#!/bin/bash

#tmux new -s lucopti

# First argument is the notebook you would like to run
notebook=$1
scriptname="$(basename $notebook .ipynb)".py

jupyter nbconvert --to script ${notebook} # --execute  && python -W ignore ${scriptname} —minloglevel=3 && rm ${scriptname}
#| grep -v 'I tensorflow/'


#python my_file
#tmux detach

