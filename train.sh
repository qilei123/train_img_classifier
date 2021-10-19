DATA_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/
OUTPUT_DIR=/data2/qilei_chen/DATA/gastro_position_clasification/work_dir
python transfer_learning.py -m ViT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 32
python transfer_learning.py -m ViT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 32 -is True
python transfer_learning.py -m Nest -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4
python transfer_learning.py -m Nest -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True
python transfer_learning.py -m TwinsSVT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4
python transfer_learning.py -m TwinsSVT -d $DATA_DIR -o $OUTPUT_DIR -c 12 -b 4 -is True