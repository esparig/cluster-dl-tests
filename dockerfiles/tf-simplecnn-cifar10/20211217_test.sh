#!/usr/bin/env bash

for batch_size in 64 128 254 512 1024 2048
do
    echo "Executing: /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=10 &"
    NOW=$(date +"%d-%m-%y_%T")
    /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=10 2>&1 | tee /data/test_${batch_size}_None_${NOW}.out
        processID=$!
        wait $processID
        sleep 30 &
    for mem_limit in 16384 8192 4096 2048 1024 512 256 128
    do
        echo "Executing: /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=10 --mem_limit=$mem_limit &"
        NOW=$(date +"%d-%m-%y_%T")
        /usr/bin/python3 cnn_cifar10.py --batch_size=$batch_size --epochs=10 --mem_limit=$mem_limit 2>&1 | tee /data/test_${batch_size}_${mem_limit}_${NOW}.out
        processID=$!
        wait $processID
        sleep 30 &
    done
done
