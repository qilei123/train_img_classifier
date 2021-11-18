export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/data3/qilei_chen/DATA/huimangban_3cls/
export OUTPUT_DIR=/data3/qilei_chen/DATA/huimangban_3cls/work_dir #_balanced
python transfer_learning.py -m mobilenetv2 -d $DATA_DIR -o $OUTPUT_DIR -c 3 -b 64 #-is True