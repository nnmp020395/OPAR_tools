#!bin/bash
jobdatefolder=$(date +'%Y%m%d')
cd '/homedata/nmpnguyen/jobs/'
mkdir ${jobdatefolder}
cd '/homedata/nmpnguyen/jobs/${jobdatefolder}'
touch log.txt
