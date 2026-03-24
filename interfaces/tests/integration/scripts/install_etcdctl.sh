#!/bin/bash

sudo apt install wget -y
mkdir -p /tmp/etcd_install
if [ "$(uname -m)" = "aarch64" ]; then
    wget https://github.com/etcd-io/etcd/releases/download/v3.4.35/etcd-v3.4.35-linux-arm64.tar.gz -O /tmp/etcd_install/etcd.tar.gz;
else
    wget https://github.com/etcd-io/etcd/releases/download/v3.4.35/etcd-v3.4.35-linux-amd64.tar.gz -O /tmp/etcd_install/etcd.tar.gz;
fi
tar -xvf /tmp/etcd_install/etcd.tar.gz -C /tmp/etcd_install
sudo mv /tmp/etcd_install/etcd-v3.4.35-linux-*/etcdctl /usr/local/bin
