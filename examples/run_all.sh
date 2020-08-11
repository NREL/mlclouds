#/bin/bash

# 
# Run cross validation against all sites on eagle. With 200 epochs, the full run 
# takes roughly 45 minutes.
#

for SITE in 0 1 2 3 4 5 6
do
    echo Kicking off autoxval and validating against $SITE
    sbatch single_axv.sh $SITE
    sleep 1
done
