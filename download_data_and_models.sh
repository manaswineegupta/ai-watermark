#!/bin/bash

curl -o datasets.tar.gz https://vault.cs.uwaterloo.ca/s/tzckM6g6PYo9Prn/download
curl -o pretrained_models.tar.gz.aa https://vault.cs.uwaterloo.ca/s/ecQd9W24oDwwKja/download
curl -o pretrained_models.tar.gz.ab https://vault.cs.uwaterloo.ca/s/mL6ZxC9itgc9wCR/download
curl -o pretrained_models.tar.gz.ac https://vault.cs.uwaterloo.ca/s/iffQ9GPSSqyx2Zt/download
curl -o pretrained_models.tar.gz.ad https://vault.cs.uwaterloo.ca/s/9kqjC7KAPQkK5DX/download
curl -o pretrained_models.tar.gz.ae https://vault.cs.uwaterloo.ca/s/iJ8aQHcLeojACR5/download
curl -o pretrained_models.tar.gz.af https://vault.cs.uwaterloo.ca/s/LqKMxiMXC5TkwQx/download
curl -o pretrained_models.tar.gz.ag https://vault.cs.uwaterloo.ca/s/9T5ZzYa26HqTAYB/download

tar -zxvf datasets.tar.gz
cat pretrained_models.tar.gz.* | tar xzvf -
rm -rf pretrained_models.tar.gz.*
rm -rf datasets.tar.gz
