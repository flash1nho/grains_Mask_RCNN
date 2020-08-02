#!/usr/bin/env bash
FOLDER="/home/iokunev/projects/grains/datasets/images"
WIDTH=617
HEIGHT=876
find ${FOLDER} -iname '*.jpg' -exec convert \{} -verbose -resize $WIDTHx$HEIGHT\> \{} \;
