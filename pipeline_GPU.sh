#!/bin/bash

# SCRIPT WILL SUBMIT ARRAY JOBS FOR GPU SERVER FOR DIFFERENT GROUPS

groups="None CTAS"
#"arr health demo mds language labs DI"

for group in $groups; do
for ii in {1..4..1}; do
	t1=$((($ii-1)*6+1))
	t2=$(($ii*6))
	echo "group="$group", batch "$ii" of 4, array start="$t1", stop="$t2
#	source test.sh $group
	qsub -N gpu_$group -v groups="$group" -t $t1-$t2 pipeline_HPF.sh
done
done

echo "END OF SCRIPT"
