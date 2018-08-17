#!/bin/bash
#Generates viewpoints using the scene.blend environment and generateView.py script. The script takes as argument the Dir to recursively look for OFF files. This script should have the first argument as the root to find .obj files. The second argument is the occlusion box size

basedir=$(dirname "$0")
blensor=/home/eduardo/opt/blensor/blensor.sh
$blensor -b $basedir/scene.blend -P $basedir/generateView.py -- $1 $2

