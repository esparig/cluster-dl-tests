# Run dcgm-exporter as a standalone container
## from https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html
### Prerequisites: Nvidia Driver
#### Guide from: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
The NVIDIA driver requires that the kernel headers and development packages for the running version of the kernel be installed at the time of the driver installation, as well whenever the driver is rebuilt. For example, if your system is running kernel version 4.4.0, the 4.4.0 kernel headers and development packages must also be installed.

The kernel headers and development packages for the currently running kernel can be installed with:
```
sudo apt-get install linux-headers-$(uname -r)
```

Ensure packages on the CUDA network repository have priority over the Canonical repository:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
sudo mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

Install the CUDA repository public GPG key:
```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
```

Setup the CUDA network repository:
```
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
```

Update the APT repository cache and install the driver using the cuda-drivers meta-package. Use the --no-install-recommends option for a lean driver install without any dependencies on X packages. This is particularly useful for headless installations on cloud insta
```
sudo apt-get update
sudo apt-get -y install cuda-drivers
```

[Optional] Install the Nvidia utils to have access to Nvidia SMI:
```
sudo apt install nvidia-utils-390
nvidia-smi
```

Install Docker:
```
sudo apt install docker.io
```

Install the Nvidia Container Toolkit:
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt install -y nvidia-container-toolkit
```
Create Dockerfile to use my own csv, and build it:
```
sudo docker build . -t dcgm-exporter
```

Run DCGM-Exporter in a docker container
```
sudo docker run -d --rm --gpus all --net host --cap-add SYS_ADMIN dcgm-exporter:latest
```

Retrieve metrics:
```
curl localhost:9400/metrics
```
