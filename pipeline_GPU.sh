#!/bin/bash

# SCRIPT WILL SUBMIT ARRAY JOBS FOR GPU SERVER FOR DIFFERENT GROUPS

# LOOP OVER GROUPS
#groups="None CTAS arr health demo mds language labs DI"
#for group in $groups; do
#	echo "group="$group
#done

# LOOP OVER SAMPLE SIZE
for ii in {1..4..1}; do
	t1=$((($ii-1)*6+1))
	t2=$(($ii*6))
for ndays in {3..3..1}; do
if [ $ndays -eq 15 ]; then
	echo "is 15; skipping"
else
	echo "number of days: "$ndays", batch "$ii" of 4, array start="$t1", stop="$t2
	qsub -N gpu_n_$ndays_batch_$t1 -v groups="mds arr CTAS",ndays=$ndays -t $t1-$t2 pipeline_HPF.sh
fi
done
done

echo "END OF SCRIPT"
