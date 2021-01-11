#!/bin/bash

# SCRIPT TRANSFER THE DATA PULLS TO HPF/SNOWQUEEN
loc=$(pwd | cut -d'/' -f3)

if [ $loc == "c" ]; then
  echo "we are on predator"
  dir_pulls=/mnt/d/projects/ED/pulls
elif [ $loc == "erik" ]; then
  echo "we are on snowqueen"
elif [ $loc == "largeprojects" ]; then
  echo "we are on hpf"
else
  echo "where are we?!"
  return
fi

if [ -d $dir_pulls ]; then
  echo "pulls folder exists"
  # Zip all the files
  if [ $loc == "c" ]; then
    zip -r --junk-path $dir_pulls/pulls.zip $dir_pulls/*
    scp $dir_pulls/pulls.zip erik@snowqueen.sickkids.ca:/home/erik/Documents/projects/ED/master
    scp $dir_pulls/pulls.zip edrysdale@data.ccm.sickkids.ca:/hpf/largeprojects/agoldenb/edrysdale/ED
  fi
  if [ $loc == "erik" ] || [ $loc == "largeprojects" ]; then
    # Unzip the files (temporary)
    #unzip -o ../pulls.zip -d ../
    #chmod 700 ../pulls*
    #rm -r ../pulls
  fi
else
  echo "pulls folder does not exist!"
  return
fi

echo "-------- END OF SCRIPT ---------"


