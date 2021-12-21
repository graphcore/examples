DATASETS_DIR=data

mkdir $DATASETS_DIR
cd $DATASETS_DIR

# download datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# decompress them
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar

# split VOC2007 and VOC2012
cd ..
python3 split_trainval_voc.py ${DATASETS_DIR}/VOCdevkit/VOC2007/
python3 split_trainval_voc.py ${DATASETS_DIR}/VOCdevkit/VOC2012/

# merge VOC2007 and VOC2012 annotations
mkdir ${DATASETS_DIR}/VOC_annotrainval_2007_2012
cp ${DATASETS_DIR}/VOCdevkit/VOC2007/Annotations_trainval/* ${DATASETS_DIR}/VOC_annotrainval_2007_2012/
cp ${DATASETS_DIR}/VOCdevkit/VOC2012/Annotations_trainval/* ${DATASETS_DIR}/VOC_annotrainval_2007_2012/

# merge VOC2007 and VOC2012 images
mkdir ${DATASETS_DIR}/VOC_images
cp ${DATASETS_DIR}/VOCdevkit/VOC2007/JPEGImages/* ${DATASETS_DIR}/VOC_images/
cp ${DATASETS_DIR}/VOCdevkit/VOC2012/JPEGImages/* ${DATASETS_DIR}/VOC_images/