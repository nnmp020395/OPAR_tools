#!/bin/bash

set -e

YEAR="2020"
MONTHS=("01" "02")
MAINDIR="/bdd/SIRTA/pub/basesirta/1a/ipral"

echo "$MAINDIR"/"$YEAR"
for m in ${MONTHS[@]}
do
	echo "$MAINDIR"/"$YEAR"/"$m"
	LISTFILES=($(find "$MAINDIR"/"$YEAR"/"$m" -type f -name "*.nc"))
	for ((i=0;i<${#LISTFILES[@]};i++))
	do 
		echo "${LISTFILES[i]}"
		python /home/nmpnguyen/new_codes_2021/test2022_v1.py -file "${LISTFILES[i]}" -opt 'ipral'
		python /home/nmpnguyen/new_codes_2021/test_get_simulationERA5.py -file '/home/nmpnguyen/tmp_file.nc' 
		python /home/nmpnguyen/Codes/RF_Ipral_test3.py -top 4000 -bottom 3000 -f "${LISTFILES[i]}"
	done
done

	# for lf in $LISTFILES
# python /home/nmpnguyen/new_codes_2021/test2022.py
# python /home/nmpnguyen/new_codes_2021/test2022_v1.py
# python /home/nmpnguyen/RF_Ipral_test3.py -f 