import os

def show_records(records_dir):

    records = open(records_dir)

    line = records.readline()

    counts = []
    count = 0
    while line:
        count+=1

        label = int(line[-2])

        if label>=len(counts):
            counts.append(1)
        else:
            counts[label]+=1

        line = records.readline()

    print(count)
    print(counts)

projects = {"binary/test":['disease_free', 'diseased'],
            "3categories/test":['disease_free', 'diseased_mild', 'diseased_severe'],
            "5categories/test":['cancer', 'disease_free', 'early_cancer', 'erosive', 'ulcer']}
#projects = {"binary/test1":[0,1]}
#projects = {"multilabel5/best_test":[0,1,2,3,4]}
#model_names = ["vgg11","densenet121","densenet161","inception3","mobilenetv2"]
model_names = ["vgg11_bn","densenet121","densenet161","mobilenetv2"]
for key in projects:
    for model_name in model_names:
        print(model_name)
        for label in projects[key]:
            records_dir = os.path.join("/data1/qilei_chen/DATA/gastro_v2",key,model_name,"best.model_"+str(label)+".txt")
            show_records(records_dir)
#show_records("/data1/qilei_chen/DATA/gastro/binary/vgg11/best.model_0.txt")
#show_records("/data1/qilei_chen/DATA/gastro/binary/vgg11/best.model_1.txt")