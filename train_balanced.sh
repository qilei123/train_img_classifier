export CUDA_VISIBLE_DEVICES=2
DATA_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/
OUTPUT_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/work_dir_balanced

#python transfer_learning.py -m resnet50 -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 32 -is True -ra True
python transfer_learning.py -m ViT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 128 -is True -l 0.1 -p True
#python transfer_learning.py -m Nest -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True
#python transfer_learning.py -m TwinsSVT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True
