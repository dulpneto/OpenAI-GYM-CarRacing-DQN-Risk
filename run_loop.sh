#!/bin/bash

if [ "$LAMB" == "" ]; then
    msg "You must define LAMB"
    rm $PID_FILE
    exit 1
fi

python3 train_model.py -l $LAMB > ../out_0.log

for i in {1..1000}
do
 CURRENT_EPSILON=$(cat 'CURRENT_EPSILON.txt');
 CURRENT_MODEL=$(cat 'CURRENT_MODEL.txt'); 
 NEXT_EPISODE=$(cat 'NEXT_EPISODE.txt');

 python3 train_model.py -l $LAMB -s $NEXT_EPISODE -p $CURRENT_EPSILON -m $CURRENT_MODEL > ../out_$i.log
done