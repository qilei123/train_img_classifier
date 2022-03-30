export CUDA_VISIBLE_DEVICES=0
#export DATA_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/
#export OUTPUT_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/work_dir
#python transfer_learning.py -m resnet50 -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 32
#python transfer_learning.py -m ViT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 128 -l 0.1 -p True
#python transfer_learning.py -m ViT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 32 -is True
#python transfer_learning.py -m Nest -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4
#python transfer_learning.py -m Nest -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True
#python transfer_learning.py -m TwinsSVT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4
#python transfer_learning.py -m TwinsSVT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True

#new innner and outside the body
export DATA_DIR=/data/qilei/.DATASETS/Endoskop4INOUT/binary/
export OUTPUT_DIR=${DATA_DIR}work_dir

#python transfer_learning.py -m resnet152 -d $DATA_DIR -o $OUTPUT_DIR -c 2 -b 64 -l 0.1 -p True

python transfer_learning.py -m resnet50 -d $DATA_DIR -o $OUTPUT_DIR -c 2 -b 128 -l 0.1 -p True