#!/bin/bash

loc=$(pwd | cut -d'/' -f3)

if [ "$loc" == "largeprojects" ]; then
  echo "We are on HPF"
dir_pulls=/hpf/largeprojects/agoldenb/edrysdale/ED/pulls
mv $dir_pulls/labs*.csv $dir_pulls/labs
mv $dir_pulls/DI*.csv $dir_pulls/DI
mv $dir_pulls/clin*.csv $dir_pulls/triage_clin
mv $dir_pulls/triage*.csv $dir_pulls/triage_notes

else
  echo "We are on predator or snow queen"
dir_pulls=/mnt/c/Users/erik\ drysdale/Documents/projects/ED/pulls
dir_hpf="edrysdale@data.ccm.sickkids.ca:/hpf/largeprojects/agoldenb/edrysdale/ED/pulls"

# the most recent month
month=${1}
year=${2}
echo "you are transfering month = "$month", year = "$year

folds="triage_clin triage_notes DI labs"

path1="$dir_pulls"/triage_clin/clin_$month$year.csv
path2="$dir_pulls"/triage_notes/triage_notes_$month$year.csv
path3="$dir_pulls"/DI/DI_$month$year.csv
path4="$dir_pulls"/labs/labs_$month$year.csv
du -sh "$path1"
du -sh "$path2"
du -sh "$path3"
du -sh "$path4"

scp "$path1" "$path2" "$path3" "$path4" $dir_hpf
fi
