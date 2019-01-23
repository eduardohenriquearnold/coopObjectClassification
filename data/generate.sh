#!/bin/bash
#Generates viewpoints using the scene.blend environment and generateView.py script.
#The script takes 4 arguments:
#1st - dir to recursevly look for OBJ files
#2nd - occlusion size (float from 0 to 1)
#3rd - random rotation (0 or 1)
#4th - destination folder 

basedir=$(dirname "$0")
blensor=/home/eduardo/opt/blensor/blensor.sh
$blensor -b $basedir/scene.blend -P $basedir/generateView.py -- $1 $2 $3 $4
