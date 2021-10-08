# cluster-dl-tests
## scripts
### simple-mnist.py
Train a simple tensorflow (Keras) NN to recognize digits from MNIST dataset.

### resnettest/imagenet_to_gcs.py
Script to convert imagenet to tfrecords.

### resnettest/resnet50-imagenet-from-scratch.sh
Bash script to train resnet50 over imagenet from scratch.

## k8s
### tf-models
#### job_mnist.yaml
Job that runs a TF training for MNIST dataset.

#### job-ls-data.yaml
Test that volume is correctly mounted.

#### job-tf-resnet50.yaml
Job that runs a training using Resnet50 model over the Imagenet dataset.

#### job_mg_mnist.yaml
Job that runs a TF training for MNIST dataset.
