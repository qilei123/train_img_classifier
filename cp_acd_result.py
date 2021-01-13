import glob
import os

def cp_acd_result(result_records_dir,img_folder_dir,save_dir):
    record_files = glob.glob(os.path.join(result_records_dir,"best.model_*"))
    labels = ['cancer', 'disease_free', 'early_cancer', 'erosive', 'ulcer']
    for record_file in record_files:
        record_file_name = os.path.basename(record_file)
        label = record_file_name[11:-4]
        records = open(record_file)
        record = records.readline()
        while record:
            split_record = record.split(" ")
            img_file_name = split_record[0]
            label_id = split_record[1]
            print(img_file_name)
            print(label_id)
            src_img_dir = os.path.join(img_folder_dir,label,img_file_name)
            if os.path.exists(src_img_dir):
                pre_label = labels[int(label_id)]
                dst_dir = os.path.join(save_dir,label,pre_label)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                command = "cp "+src_img_dir+" "+dst_dir
                os.system(command)
            record = records.readline()
    pass

if __name__ == "__main__":
    result_records_dir = "/data1/qilei_chen/DATA/gastro_v2/5categories/test2/mobilenetv2_500/"
    img_folder_dir = "/data1/qilei_chen/DATA/gastro_v2/5categories/val/"
    save_dir = "/data1/qilei_chen/DATA/gastro_v2/5categories/test2/mobilenetv2_500/"
    cp_acd_result(result_records_dir,img_folder_dir,save_dir)