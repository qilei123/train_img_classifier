export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=/data3/qilei_chen/DATA/endoscope3/
export OUTPUT_DIR=/data3/qilei_chen/DATA/endoscope3/work_dir_balanced
python transfer_learning.py -m resnet50 -d $DATA_DIR -o $OUTPUT_DIR -c 3 -b 128 -is True