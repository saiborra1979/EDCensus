#!/bin/bash

# SCRIPT WILL SUBMIT ARRAY JOBS FOR GPU SERVER FOR DIFFERENT GROUPS

groups_final="mds arr CTAS"

# LOOP OVER SAMPLE SIZE
for ii in {1..4..1}; do
	t1=$((($ii-1)*6+1))
	t2=$(($ii*6))
for ndays in {3..3..1}; do
	echo "number of days: "$ndays", batch "$ii" of 4, array start="$t1", stop="$t2
	qsub -N gpu_n_$ndays_batch_$t1 -v groups="$groups_final",ndays=$ndays -t $t1-$t2 pipeline_HPF.sh
done
done

echo "END OF SCRIPT"
