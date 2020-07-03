#!/bin/bash
if [ ! -d "thirdparty" ] ; then
    echo "Cloning data generator code"
    git clone --depth=1 https://github.com/datalogue/keras-attention thirdparty
    mkdir data
    sed 's/fake.seed/Faker.seed/' thirdparty/data/generate.py > data/generate.py
    sed -i "s/len(int2machine)+1:'<eot>'})/len(int2machine)+1:'<eot>', len(int2machine)+2: '<sot>'})/" data/generate.py
    sed 's/from keras/# from keras/' thirdparty/data/reader.py > data/reader.py
fi
python3 data/generate.py
