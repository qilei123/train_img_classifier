import os
def create_dataset_with_records(records_dir,images_dir,dataset_dir):
    records = open(records_dir)

    line = records.readline()
    label_to_folder = {0:"Disease",
                        1:"NotDisease",
                        2:"Unqualified"}
    while line:
        line_splits = line.split(" ")
        
        img_dir = os.path.join(images_dir,line_splits[0].replace("/home/zys/data/C3R/Integrated/",""))
        
        dst_dir = os.path.join(dataset_dir,label_to_folder[int(line_splits[1])])
        #print(dst_dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        #print(img_dir)
        command = "cp "+images_dir + " "+dst_dir
        print(command)
        os.system(command)

        line = records.readline()

create_dataset_with_records("/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/train.txt",
                            "/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/",
                            "/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/train/")

create_dataset_with_records("/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/val.txt",
                            "/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/",
                            "/data1/qilei_chen/DATA/ROP_DATASET/ROP_201707_C3R_Integrated/Integrated/val/")