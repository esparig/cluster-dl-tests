FROM nvcr.io/nvidia/k8s/dcgm-exporter:2.2.9-2.5.0-ubi8

COPY dcp-metrics-included.csv /etc/dcgm-exporter/default-counters.csv

ENTRYPOINT /usr/local/dcgm/dcgm-exporter-entrypoint.sh
