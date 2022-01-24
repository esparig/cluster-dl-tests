#!/usr/bin/env bash

NOW=$(date +"%d-%m-%y_%T")
/usr/bin/python3 cnn_cifar10.py --model_summary --epochs=0 2>&1 | tee /data/test_model_summary_${NOW}.out

for batch_size in 64 128 256
do
    for mem_limit in 12288 8192 4096 2048 1024
    do
        echo "Executing: /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=20 --mem_limit=$mem_limit &"
        
        /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=20 --mem_limit=$mem_limit 2>&1 | tee /data/test_${batch_size}_${mem_limit}_${NOW}.out
        processID=$!
        wait $processID
        sleep 30 &
    done
done
