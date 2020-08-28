#!/bin/bash

# SCRIPT TRANSFER THE DATA PULLS TO HPF/SNOWQUEEN
dir_pulls=../pulls

is_mnt=$(pwd | cut -d'/' -f3)

if [ $is_mnt == "c" ]; then
  echo "we are on predator"
elif [ $is_mnt == "erik" ]; then
  echo "we are on snowqueen"
elif [ $is_mnt == "edrysdale" ]; then
  echo "we are on hpf"
else
  echo "where are we?!"
  return
fi

if [ -d $dir_pulls ]; then
  echo "pulls folder exists"
  # Zip all the files
  if [ $is_mnt == "c" ]; then
    zip -r ../pulls.zip $dir_pulls
    scp ../pulls.zip erik@snowqueen.sickkids.ca:/home/erik/Documents/projects/ED/master
    scp ../pulls.zip edrysdale@data.ccm.sickkids.ca:/hpf/largeprojects/agoldenb/edrysdale/ED
  fi
  if [ $is_mnt == "erik" ] || [ $is_mnt == "edrysdale" ]; then
    rm -r ../pulls
    unzip -o ../pulls.zip -d ../
    chmod 700 ../pulls*
  fi
else
  echo "pulls folder does not exist!"
  return
fi

echo "-------- END OF SCRIPT ---------"


