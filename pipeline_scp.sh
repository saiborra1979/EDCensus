#!/bin/bash

# SCRIPT TRANSFER THE DATA PULLS TO HPF/SNOWQUEEN
dir_pulls=../pulls

loc=$(pwd | cut -d'/' -f3)

if [ $loc == "c" ]; then
  echo "we are on predator"
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
    zip -r ../pulls.zip $dir_pulls
    scp ../pulls.zip erik@snowqueen.sickkids.ca:/home/erik/Documents/projects/ED/master
    scp ../pulls.zip edrysdale@data.ccm.sickkids.ca:/hpf/largeprojects/agoldenb/edrysdale/ED
  fi
  if [ $loc == "erik" ] || [ $loc == "largeprojects" ]; then
    rm -r ../pulls
    unzip -o ../pulls.zip -d ../
    chmod 700 ../pulls*
  fi
else
  echo "pulls folder does not exist!"
  return
fi

echo "-------- END OF SCRIPT ---------"


