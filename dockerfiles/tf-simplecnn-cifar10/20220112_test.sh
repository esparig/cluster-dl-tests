#!/usr/bin/env bash

/usr/bin/python3 cnn_cifar10.py --model_summary --epochs=0 2>&1 | tee /data/test_model_summary_${NOW}.out

for batch_size in 64 128 256
do
    echo "Executing: /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=100 &"
    NOW=$(date +"%d-%m-%y_%T")
    /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=100 2>&1 | tee /data/test_${batch_size}_None_${NOW}.out
        processID=$!
        wait $processID
        sleep 30 &
    for mem_limit in 16384 8192 4096 2048 1024
    do
        echo "Executing: /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=100 --mem_limit=$mem_limit &"
        NOW=$(date +"%d-%m-%y_%T")
        /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=100 --mem_limit=$mem_limit 2>&1 | tee /data/test_${batch_size}_${mem_limit}_${NOW}.out
        processID=$!
        wait $processID
        sleep 30 &
    done
done
