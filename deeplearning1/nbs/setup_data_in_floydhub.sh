#!/bin/sh

mkdir -p /output/models/glove/
cd /output/models/
wget -c --show-progress http://files.fast.ai/models/imagenet_class_index.json
wget -c --show-progress http://files.fast.ai/models/resnet50.h5
wget -c --show-progress http://files.fast.ai/models/resnet_nt.h5
wget -c --show-progress http://files.fast.ai/models/vgg16.h5
wget -c --show-progress http://files.fast.ai/models/vgg16_bn.h5
wget -c --show-progress http://files.fast.ai/models/vgg16_bn_conv.h5

cd /output/models/glove/
wget -c --show-progress http://files.fast.ai/models/glove/6B.50d.tgz
tar -xzf 6B.50d.tgz

cd /output/
wget -c --show-progress http://files.fast.ai/files/dogscats.zip
unzip dogscats.zip

rm dogscats.zip
rm /output/models/glove/6B.50d.tgz

global_path="/output/dogscats/"
mkdir -p $global_path/test1/unknown/
mv $global_path/test1/*.jpg $global_path/test1/unknown/

sample_path="/output/dogscats/sample/"
mkdir -p $sample_path/test1/unknown/
cp $global_path/test1/unknown/4*09.jpg $sample_path/test1/unknown/