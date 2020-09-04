#!/bin/bash

# SCRIPT WILL SUBMIT ARRAY JOBS FOR GPU SERVER FOR DIFFERENT GROUPS

groups="CTAS arr health demo mds language labs DI"

for group in $groups; do
	echo $group
	qsub -N gpu_$group -v groups="$group" pipeline_HPF.sh
done

echo "END OF SCRIPT"
