
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import time
import copy
import os
import sys
import argparse
import csv
#sys.path.insert(0,'/media/cql/DATA1/Development/vision2')
#sys.path.insert(0,'/data0/qilei_chen/Development/vision2')
#sys.path.insert(0,'/data1/qilei_chen/DEVELOPMENTS/vision2')
import torchvision
from torchvision import datasets, models, transforms
#from networks import *
from torch.autograd import Variable
from PIL import Image
import glob
import cv2
import datetime
import time
def micros(t1, t2):
    delta = (t2-t1).microseconds
    return delta

from img_crop import crop_img

import threading
class classifier:
    def __init__(self,input_size_=1000,mean_=[0.485, 0.456, 0.406],std_=[0.229, 0.224, 0.225],class_num_=2,
                model_name = 'resnet101_wide',device_id=0,with_gray=False):
        self.input_size = input_size_
        self.mean = mean_
        self.std = std_
        if with_gray:
            self.test_transform = transforms.Compose([
                    transforms.Resize((self.input_size,self.input_size)),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        else:
            self.test_transform = transforms.Compose([
                    transforms.Resize((self.input_size,self.input_size)),
                    #transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        self.class_num = class_num_
        self.device = torch.device("cuda:"+str(device_id))
        if model_name == "alexnet":
            """ Alexnet
            """
            self.model = models.alexnet()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224

        elif model_name == "vgg11_bn":
            """ VGG11_bn
            """
            self.model = models.vgg11_bn()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg11":
            """ VGG11
            """
            self.model = models.vgg11()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg13_bn":
            """ VGG13_bn
            """
            self.model = models.vgg13_bn()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg13":
            """ VGG13
            """
            self.model = models.vgg13()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg16_bn":
            """ VGG16_bn
            """
            self.model = models.vgg16_bn()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg16":
            """ VGG16
            """
            self.model = models.vgg16()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg19_bn":
            """ VGG19_bn
            """
            self.model = models.vgg19_bn()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg19":
            """ VGG19
            """
            self.model = models.vgg19()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "squeezenet1_0":
            """ squeezenet1_0
            """
            self.model = models.squeezenet1_0()
            #set_parameter_requires_grad(model_ft, feature_extract)
            self.model.classifier[1] = nn.Conv2d(512, self.class_num, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.class_num
            input_size = 224
        elif model_name == "squeezenet1_1":
            """ squeezenet1_1
            """
            self.model = models.squeezenet1_1()
            #set_parameter_requires_grad(model_ft, feature_extract)
            self.model.classifier[1] = nn.Conv2d(512, self.class_num, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.class_num
            input_size = 224
        elif model_name == "resnet18":
            """ Resnet18
            """
            self.model = models.resnet18()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet34":
            """ Resnet34
            """
            self.model = models.resnet34()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet50":
            """ Resnet50
            """
            self.model = models.resnet50()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet101":
            """ Resnet101
            """
            self.model = models.resnet101()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet152":
            """ Resnet152
            """
            self.model = models.resnet152()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name=='inception3':
            self.model = models.inception_v3()
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name=='inception_v3_wide':
            self.model = models.inception_v3_wide()
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name=='resnet101_wide':
            self.model = models.resnet101_wide()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name == "densenet121":
            """ Densenet
            """
            self.model = models.densenet121()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
        elif model_name == "densenet161":
            self.model = models.densenet161()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num)
        elif model_name == "densenet169":
            self.model = models.densenet169()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
        elif model_name == "densenet201":
            self.model = models.densenet201()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num)
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs,self.class_num)
            #input_size = 224  
        elif model_name == "shufflenetv2_x0_5":
            self.model = models.shufflenetv2_x0_5()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            #input_size = 224  
        elif model_name == "mobilenetv2":
            self.model = models.mobilenet_v2()
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs,self.class_num)
            #input_size = 224          
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def ini_model(self,model_dir):
        checkpoint = torch.load(model_dir,map_location = self.device)
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint)
        
        #self.model.cuda()
        self.model.to(self.device)
        #print(self.model)
        cudnn.benchmark = True
        self.model.eval()
    def predict(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image = Image.open(img_dir).convert('RGB')
        image = Image.fromarray(img)
        
        image = self.test_transform(image)
        inputs = image
        inputs = Variable(inputs)
        
        inputs = inputs.to(self.device)
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front
        t1 = datetime.datetime.now()
        outputs = self.model(inputs)
        t2 = datetime.datetime.now()
        _, preds = torch.max(outputs, 1)
        #print(preds)
        softmax_res = self.softmax(outputs.data.cpu().numpy()[0])
        probilities = []
        for probility in softmax_res:
            probilities.append(probility)
        #print(probilities)
        
        #print(micros(t1,t2)/1000)
        #return probilities.index(max(probilities)),probilities
        return probilities.index(max(probilities))
    def predict_batch(self,img_batch):
        batch = []
        for img in img_batch:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #image = Image.open(img_dir).convert('RGB')
            image = Image.fromarray(img)
            image = self.test_transform(image)
            batch.append(image)
        inputs = Variable(torch.stack(batch))
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        outputs_gpu = outputs.data.cpu().numpy()
        labels = []
        probs = []
        for i in range(len(outputs_gpu)):
            softmax_res = self.softmax(outputs_gpu[i])
            probs.append(softmax_res)
            labels.append(np.where(softmax_res==max(softmax_res)))
        return labels,probs
            

    def predict1(self,img_dir):
        img = cv2.imread(img_dir)
        return self.predict(img)

def process_4_situation_videos(model_name = "densenet161"):
    
    print("start ini model")
    model = classifier(224,model_name=model_name,class_num_=4)

    #model1 = classifier(224,model_name=model_name,class_num_=4,device_id=1)

    model_dir = '/data2/qilei_chen/DATA/GI_4_NEW/finetune_4_new_oimg_'+model_name+'/best.model'

    model.ini_model(model_dir)
    print("finish ini model")
    #model1.ini_model(model_dir)

    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/weijingshi4/"
    videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
    '''
    big_roi = [441, 1, 1278, 720]
    small_roi = [156, 40, 698, 527]

    roi = big_roi
    '''
    video_start = -1#15

    videos_result_folder = os.path.join(videos_folder,"result_"+model_name)

    video_suffix = ".avi"
    
    video_file_dir_list = glob.glob(os.path.join(videos_folder,"*"+video_suffix))
    #print(video_file_dir_list)
    #return
    if not os.path.exists(videos_result_folder):
        os.makedirs(videos_result_folder)
    video_count=0
    for video_file_dir in video_file_dir_list:

        if video_count>video_start:
            count=1

            video = cv2.VideoCapture(video_file_dir)

            success,frame = video.read()
        
            video_name = os.path.basename(video_file_dir)

            records_file_dir = os.path.join(videos_result_folder,video_name.replace(video_suffix,".txt"))
            #records_file_header = open(records_file_dir,"w")

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            show_result_video_dir = os.path.join(videos_result_folder,video_name)
            #videoWriter = cv2.VideoWriter(show_result_video_dir,cv2.VideoWriter_fourcc("P", "I", "M", "1"),fps,frame_size)
            print(show_result_video_dir)
            while success:
                '''
                frame_roi = frame[roi[1]:roi[3],roi[0]:roi[2]]
                predict_label = model.predict(frame_roi)
                '''
                predict_label = model.predict(frame)
                #predict_label1 = model1.predict(frame)
                #records_file_header.write(str(count)+" "+str(predict_label)+"\n")
                #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame_roi)
                #cv2.putText(frame,str(count)+":"+str(predict_label),(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
                #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame)
                #videoWriter.write(frame)
                #print(predict_label)
                success,frame = video.read()
                count+=1
                '''
                if count%10000==0:
                    print(count)
                '''
        video_count+=1

'''
model_name='densenet121'
cf = classifier(224,model_name=model_name,class_num_=4)
#lesion_category = 'Cotton_Wool_Spot'
folder_label = 3
#model_dir = '/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/models_4_'+lesion_category+'/densenet_epoch_16.pth'
model_dir = '/data2/qilei_chen/DATA/4class_c/finetune_4_end0_'+model_name+'/best.model'
cf.ini_model(model_dir)
#for i in range(100):
#image_file_dirs = glob.glob('/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/val/'+str(folder_label)+'/*.jpg')
image_file_dirs = glob.glob('/data2/qilei_chen/DATA/4class_c/val/'+str(folder_label)+'/*.jpg')
#print(image_file_dirs)
#count = 0
wrong_count=0
count = [0,0,0,0,0]
print('groundtruth:'+str(folder_label))
for image_file_dir in image_file_dirs:
    #print(image_file_dir)
    label = cf.predict1(image_file_dir)
    
    if label!=folder_label:
        print(label)
        print(image_file_dir)
        pass
    #'
    #    wrong_count+=1
    #    #cv2.imshow('test',cv2.imread(image_file_dir))
    #    #cv2.waitKey(0)
    #count += 1
    #'
    count[int(label)]+=1
print(count)
'''

'''
print(cf.predict('/home/cql/Downloads/test5.7/test/16_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/172_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/217_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/286_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/508_left.jpeg'))

print(cf.predict('/home/cql/Downloads/test5.7/test0/13_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/22_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/31_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/40_right.jpeg'))
'''

#process_4_situation_videos(model_name='alexnet')
#process_4_situation_videos(model_name='squeezenet1_0')
#process_4_situation_videos(model_name='squeezenet1_1')
#process_4_situation_videos(model_name='inception3')

#process_4_situation_videos(model_name='vgg11')
#process_4_situation_videos(model_name='vgg13')
#process_4_situation_videos(model_name='vgg16')
#process_4_situation_videos(model_name='vgg19')

#process_4_situation_videos(model_name='vgg11_bn')
#process_4_situation_videos(model_name='vgg13_bn')
#process_4_situation_videos(model_name='vgg16_bn')
#process_4_situation_videos(model_name='vgg19_bn')

#process_4_situation_videos(model_name='densenet121')
#process_4_situation_videos(model_name='densenet161')
#process_4_situation_videos(model_name='densenet169')
#process_4_situation_videos(model_name='densenet201')

#process_4_situation_videos(model_name='resnet18')
#process_4_situation_videos(model_name='resnet34')
#process_4_situation_videos(model_name='resnet50')
#process_4_situation_videos(model_name='resnet101')
#process_4_situation_videos(model_name='resnet152')

import pandas as pd
import numpy as np

xray_model_names = ["squeezenet1_0","shufflenetv2_x0_5","mobilenet_v2"]
with_model = False
def test_4_xray(model_name=xray_model_names[0],folder_id=0):
    
    print("start ini model")
    model = classifier(224,model_name=model_name,class_num_=2)
    #model1 = classifier(224,model_name=model_name,class_num_=4,device_id=1)
    model_dir = '/data2/qilei_chen/DATA/xray/balanced_finetune_2_'+model_name+'/'+str(folder_id)+'_best.model'
    model.ini_model(model_dir)
    print("finish ini model")

    test_dataset_folder_dir = "/data1/qilei_chen/DATA/CheXpert/SUBSETS-small/"
    test_dataset_anno = os.path.join(test_dataset_folder_dir,"combined.csv")
    
    pd_frame = pd.read_csv(test_dataset_anno,sep=',')

    file_dirs = pd_frame.filename.to_numpy()
    labels = pd_frame.Abnormal

    result_records=[]
    counts = [0,0]
    error_counts = [0,0]

    result_records_file = open(os.path.join(test_dataset_folder_dir,model_name+"_records.txt"),'w')

    for file_dir,label in zip(file_dirs,labels):

        img_dir = os.path.join(test_dataset_folder_dir,file_dir)

        result_records.append(model.predict1(img_dir))

        result_records_file.write(str(result_records[-1])+"\n")

        counts[int(label)]+=1

        if label==result_records[-1]:
            error_counts[int(label)]+=1

        if sum(counts)%1000==0:
            print(str(sum(counts))+"...")

    print(counts)
    print(error_counts)
#test_4_xray(model_name=xray_model_names[0])
#test_4_xray(model_name=xray_model_names[1])
#test_4_xray(model_name=xray_model_names[2])
'''
# this part for DB
model = classifier(224,model_name="squeezenet1_0",class_num_=3)
#model1 = classifier(224,model_name=model_name,class_num_=4,device_id=1)
#model_dir = "/data1/qilei_chen/DATA/DB_NATURAL/data3/squeezenet1_0/best.model"
model_dir = "/home/cql/Downloads/best.model"
model.ini_model(model_dir)
#print(model.predict1("/data1/qilei_chen/DEVELOPMENTS/train_4_torch_vision/XYNFMK202007230003_L_ORG.jpg"))
import glob
test_files = glob.glob("/media/cql/DATA0/baidupan/2020_xiangyaneifenmike/xiangyaneifenmiyanke_20020804result/*.jpg")
for test_file in test_files:
    if "ORG" in test_file:
        print(os.path.basename(test_file))
        print(model.predict1(test_file))
'''
def test_4_gastro(img_dir,model_name,model_dir,label,class_num,input_size):
    if "inception" in model_name:
        model = classifier(299,model_name=model_name,class_num_=class_num)
    else:
        #input_size = input_size
        #print(input_size)
        model = classifier(input_size,model_name=model_name[:-4],class_num_=class_num)
    
    model.ini_model(model_dir)

    #records = open(model_dir+"_"+str(label)+".txt",'w')

    img_files = glob.glob(os.path.join(img_dir,str(label),"*.jpg"))
    print(os.path.join(img_dir,str(label),"*.jpg"))
    for img_file in img_files:
        prelabel,probs = model.predict1(img_file)
        #print(img_file)
        #print(prelabel)
        img_name = os.path.basename(img_file)
        #records.write(img_name+" "+str(prelabel)+" "+str(probs)+"\n")
'''        
img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "vgg11"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/vgg11/best.model"
label = 0
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)

img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "vgg11"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/vgg11/best.model"
label = 1
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)

img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "densenet121"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/densenet121/best.model"
label = 0
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)

img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "densenet121"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/densenet121/best.model"
label = 1
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)

img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "inception3"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/inception3/best.model"
label = 0
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)

img_dir = "/data1/qilei_chen/DATA/gastro/binary/val/"
model_name = "inception3"
model_dir = "/data1/qilei_chen/DATA/gastro/binary/inception3/best.model"
label = 1
class_num = 2
test_4_gastro(img_dir,model_name,model_dir,label,class_num)
'''
'''
model_names = ['vgg11','densenet121','inception3']
labels = [0,1,2,3,4,5]
img_dir = "/data1/qilei_chen/DATA/gastro/multilabel/val/"
class_num = 6
for model_name in model_names:
    model_dir = "/data1/qilei_chen/DATA/gastro/multilabel/"+model_name+"/best.model"
    for label in labels:
        test_4_gastro(img_dir,model_name,model_dir,label,class_num)
'''

'''
model_names = ["resnet50_500","vgg11_500","densenet121_500","densenet161_500","mobilenetv2_500",]
#model_names = ['mobilenetv2']

datasets = {"binary":['disease_free', 'diseased'],
            "3categories":['disease_free', 'diseased_mild', 'diseased_severe'],
            "5categories":['cancer', 'disease_free', 'early_cancer', 'erosive', 'ulcer']}

for key in datasets:
    labels = datasets[key]
    class_num = len(labels)
    img_dir = "/data1/qilei_chen/DATA/gastro_v2/"+key+"/val/"
    for model_name in model_names:
        model_dir = "/data1/qilei_chen/DATA/gastro_v2/"+key+"/test2/"+model_name+"/best.model"
        for label in labels:
            print(key)
            print(model_name)
            input_size = 500
            print(input_size)
            test_4_gastro(img_dir,model_name,model_dir,label,class_num,input_size)
'''

def process_4_situation_videos_gray(videos_folder,model_dir,model_name ,videos_result_folder,class_num = 5,selected_videos = "*_w*"):
    os.system("export OMP_NUM_THREADS=4")
    print("start ini model")
    model = classifier(224,model_name=model_name,class_num_=class_num)

    #model1 = classifier(224,model_name=model_name,class_num_=4,device_id=1)

    #model_dir = '/data2/qilei_chen/DATA/GI_4_NEW_GRAY/finetune_4_new_oimg_'+model_name+'/best.model'

    model.ini_model(model_dir)
    print("finish ini model")
    #model1.ini_model(model_dir)

    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/weijingshi4/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
    '''
    big_roi = [441, 1, 1278, 720]
    small_roi = [156, 40, 698, 527]

    roi = big_roi
    '''
    video_start = -1#15

    #video_suffix1 = ".avi"
    
    #video_file_dir_list = glob.glob(os.path.join(videos_folder,"*"+video_suffix))

    #video_suffix = ".mp4"
    
    video_file_dir_list = glob.glob(os.path.join(videos_folder,"videos",selected_videos))
    #print(video_file_dir_list)
    #return
    if not os.path.exists(videos_result_folder):
        os.makedirs(videos_result_folder)
    video_count=0
    for video_file_dir in video_file_dir_list:
        if ('_w' in video_file_dir) or True:
            if video_count>video_start:
                print(video_file_dir)
                count=1

                video = cv2.VideoCapture(video_file_dir)

                success,frame = video.read()
            
                video_name = os.path.basename(video_file_dir)

                records_file_dir = os.path.join(videos_result_folder,video_name[:-4]+".txt")
                print(records_file_dir)
                if os.path.exists(records_file_dir):
                    records_file_header = open(records_file_dir)
                    content = records_file_header.readline()
                    if content=='':
                        pass
                    else:
                        continue

                records_file_header = open(records_file_dir,"w")

                fps = video.get(cv2.CAP_PROP_FPS)
                frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                #show_result_video_dir = os.path.join(videos_result_folder,video_name)
                #videoWriter = cv2.VideoWriter(show_result_video_dir,cv2.VideoWriter_fourcc("P", "I", "M", "1"),fps,frame_size)
                #print(show_result_video_dir)
                while success:
                    '''
                    frame_roi = frame[roi[1]:roi[3],roi[0]:roi[2]]
                    predict_label = model.predict(frame_roi)
                    '''
                    predict_label = model.predict(frame)
                    #predict_label1 = model1.predict(frame)
                    records_file_header.write(str(count)+" "+str(predict_label)+"\n")
                    #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame_roi)
                    #cv2.putText(frame,str(count)+":"+str(predict_label),(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
                    #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame)
                    #videoWriter.write(frame)
                    #print(predict_label)
                    success,frame = video.read()
                    count+=1
                    '''
                    if count%10000==0:
                        print(count)
                    '''
        
        video_count+=1


def process_4_situation_videos_gray_batch(videos_folder,model_dir,model_name ,videos_result_folder,class_num = 5,selected_videos = "*_w*",batch_size=8):
    #os.system("export OMP_NUM_THREADS=4")
    print("start ini model")
    model = classifier(224,model_name=model_name,class_num_=class_num)

    #model1 = classifier(224,model_name=model_name,class_num_=4,device_id=1)

    #model_dir = '/data2/qilei_chen/DATA/GI_4_NEW_GRAY/finetune_4_new_oimg_'+model_name+'/best.model'

    model.ini_model(model_dir)
    print("finish ini model")
    #model1.ini_model(model_dir)

    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/weijingshi4/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
    '''
    big_roi = [441, 1, 1278, 720]
    small_roi = [156, 40, 698, 527]

    roi = big_roi
    '''
    video_start = -1#15

    #video_suffix1 = ".avi"
    
    #video_file_dir_list = glob.glob(os.path.join(videos_folder,"*"+video_suffix))

    #video_suffix = ".mp4"
    
    video_file_dir_list = glob.glob(os.path.join(videos_folder,"videos",selected_videos))
    #print(video_file_dir_list)
    #return
    if not os.path.exists(videos_result_folder):
        os.makedirs(videos_result_folder)
    video_count=0
    for video_file_dir in video_file_dir_list:
        if ('_w' in video_file_dir) or True:
            if video_count>video_start:
                print(video_file_dir)
                count=1

                video = cv2.VideoCapture(video_file_dir)

                success,frame = video.read()
            
                video_name = os.path.basename(video_file_dir)

                records_file_dir = os.path.join(videos_result_folder,video_name[:-4]+".txt")
                print(records_file_dir)
                if os.path.exists(records_file_dir):
                    records_file_header = open(records_file_dir)
                    content = records_file_header.readline()
                    if content=='':
                        pass
                    else:
                        continue

                records_file_header = open(records_file_dir,"w")

                fps = video.get(cv2.CAP_PROP_FPS)
                frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                #show_result_video_dir = os.path.join(videos_result_folder,video_name)
                #videoWriter = cv2.VideoWriter(show_result_video_dir,cv2.VideoWriter_fourcc("P", "I", "M", "1"),fps,frame_size)
                #print(show_result_video_dir)
                img_batch = []
                frame_indexes = []
                while success:
                    '''
                    frame_roi = frame[roi[1]:roi[3],roi[0]:roi[2]]
                    predict_label = model.predict(frame_roi)
                    '''
                    if len(img_batch)>=batch_size:

                        #predict_label = model.predict(frame)
                        predict_results = model.predict_batch(img_batch)
                        #predict_label1 = model1.predict(frame)
                        for frame_index,predict_label,predict_probs in zip(frame_indexes,predict_results[0],predict_results[1]):

                            records_file_header.write(str(frame_index)+" #"+str(predict_label[0][0])+"# "+str(predict_probs)+"\n")
                        img_batch = []
                        frame_indexes = []
                        #print(count)
                    else:
                        img_batch.append(frame)
                        frame_indexes.append(count)
                    #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame_roi)
                    #cv2.putText(frame,str(count)+":"+str(predict_label),(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
                    #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame)
                    #videoWriter.write(frame)
                    #print(predict_label)
                    success,frame = video.read()
                    count+=1
                    
                    #if count==10:
                    #    break
                    
        #break
        video_count+=1

import threading

def test_db_quality(root,records_dir,model_dir):
    records_file = open(os.path.join(root,records_dir))

    model = classifier(input_size_=224,class_num_=3,model_name='squeezenet1_0')
    model.ini_model(model_dir)

    file_dir = records_file.readline()

    quality_records = open(os.path.join(root,records_dir+".quality"),"w")

    while file_dir:
        if "ORG" in file_dir:
            file_dir = file_dir[:-1]
            quality_label = model.predict1(os.path.join(root,file_dir))
            quality_records.write(file_dir+" "+str(quality_label)+"\n")
        file_dir = records_file.readline()

def show_confusion_matrix(model,folder_dir,result_dir,label_list):
    
    for label in label_list:
        record_file = open(os.path.join(result_dir,str(label)+".txt"),'w')
        img_list = glob.glob(os.path.join(folder_dir,str(label),'*.jpg'))
        count = [0]*(len(label_list))
        for img_dir in img_list:
            image = cv2.imread(img_dir)
            predict_label = model.predict(image)
            img_name = os.path.basename(img_dir)
            record_file.write(img_name+" "+str(predict_label)+"\n")
            
            count[int(predict_label[0])]+=1

        print(count)

def create_confusion_matrix():
    model_name="mobilenetv2"
    data_set = "5class_scene_combine_2_fine_2_3"
    data_set = '5class_scene_combine_2_fine_2_3_fine_c_in'
    #data_set = '5class_scene_combine_2_fine_2_3_fine_c_in_fine_out'
    labels = [0,1,2,3,4]
    model_dir = "/data2/qilei_chen/DATA/"+data_set+"/work_dir/mobilenetv2_2/best.model"
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(model_dir)
    show_confusion_matrix(model,"/data2/qilei_chen/DATA/"+data_set+"/val",labels)

def test_videos():
    model_name="mobilenetv2"
    dataset_name = "5class_scene_alex_manual"
    dataset_name = "5class_scene_combine_2_fine_2_3"
    dataset_name = '5class_scene_combine_2_fine_2_3_fine_c_in'
    dataset_name = '5class_scene_combine_2_fine_2_3_fine_c_in_fine_out'
    dataset_name = '5class_scene_combine_2_fine_2_3_fine_c_in_fine_out_combine_train_val'
    model_dir = "/data2/qilei_chen/DATA/"+dataset_name+"/work_dir/mobilenetv2_2/best.model"
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_all/"
    #selected_videos = "*_w*"
    selected_videos = "*_c*"
    selected_videos = "*"
    use_batch=''
    use_batch = "_batch_66"
    videos_result_folder = os.path.join(videos_folder_dir,dataset_name+"_"+model_name+use_batch)
    #process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder,selected_videos=selected_videos)
    process_4_situation_videos_gray_batch(videos_folder_dir,model_dir,
                                            model_name,videos_result_folder,
                                            selected_videos=selected_videos,batch_size=126)
def test_batch():
    model_name="mobilenetv2"
    dataset_name = "5class_scene_alex_manual"
    dataset_name = "5class_scene_combine_2_fine_2_3"
    dataset_name = '5class_scene_combine_2_fine_2_3_fine_c_in'
    model_dir = "/data2/qilei_chen/DATA/"+dataset_name+"/work_dir/mobilenetv2_2/best.model"
    labels = [0,1,2,3,4]
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(model_dir) 
    image1 = cv2.imread("/data2/qilei_chen/DATA/5class_scene_alex_manual/20191011_1611_1632_c_14074.jpg")
    image2 = cv2.imread("/data2/qilei_chen/DATA/5class_scene_alex_manual/20191011_1611_1632_c_3890.jpg")
    img_batch = [image1,image2]
    print(model.predict_batch(img_batch))       

def create_confusion_matrix_endoscope3():
    model_name="resnet50"
    labels = ['colonoscopy', 'extracorporal', 'gastroscopy']
    model_dir = "/data3/qilei_chen/DATA/endoscope3/work_dir/resnet50_2/best.model"
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(model_dir)
    show_confusion_matrix(model,
        "/data3/qilei_chen/DATA/endoscope3/val",
        "/data3/qilei_chen/DATA/endoscope3/work_dir/resnet50_2",
        labels)
def test_on_videos_endoscope3():
    src_dir = "/data2/DB_GI/processed_videos"
    results_dir = "/data2/DB_GI/processed_videos_result"

    video_dirs = glob.glob(os.path.join(src_dir,"*.avi"))
    
    model_name="resnet50"
    labels = ['colonoscopy', 'extracorporal', 'gastroscopy']
    model_dir = "/data3/qilei_chen/DATA/endoscope3/work_dir/resnet50_2/best.model"
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(model_dir)    

    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        result_dir = os.path.join(results_dir,video_name+".txt")

        result_file = open(result_dir,'w')

        video_reader = cv2.VideoCapture(video_dir)
        success,frame = video_reader.read()
        frame_index=1
        while success:
            predict_label = model.predict(frame)
            result_file.write(str(frame_index)+" #"+str(predict_label[0])+"\n")
            frame_index+=1
            success,frame = video_reader.read()     

def create_confusion_matrix_huimangban():
    model_names=["resnet50","mobilenetv2"]
    folder_names = ["huimangban_3cls","huimangban_3cls_cropped"]
    balances = ["","_balanced"]
    labels = ['0', '1', '2']
    for balance in balances:
        for folder_name in folder_names:
            for model_name in model_names:
                model_dir = "/data3/qilei_chen/DATA/"+folder_name+"/work_dir"+balance+"/"+model_name+"_2/best.model"
                model = classifier(224,model_name=model_name,class_num_=len(labels))
                model.ini_model(model_dir)
                print(balance+"-------"+folder_name+"---------"+model_name)
                show_confusion_matrix(model,
                    "/data3/qilei_chen/DATA/"+folder_name+"/val",
                    "/data3/qilei_chen/DATA/"+folder_name+"/work_dir"+balance+"/"+model_name+"_2",
                    labels)
def test_on_videos_huimangban():
    record_file ="/data3/qilei_chen/DATA/hmb_gt.txt"
    rf = open(record_file)
    line = rf.readline()
    model_name = "mobilenetv2"
    labels = ['0', '1', '2']
    model_dir = "/data3/qilei_chen/DATA/huimangban_3cls/work_dir_balanced/mobilenetv2_2/best.model"
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(model_dir)
    video_dir = "/data2/yli/third_c_videos_split/"
    while line:
        eles = line.split(",")
        video_name = eles[0]
        images = glob.glob(os.path.join(video_dir,video_name,"*.jpg"))
        result_file = open("/data3/qilei_chen/DATA/huimangban_3cls/work_dir_balanced/mobilenetv2_2/"+video_name+".txt",'w')
        for i in range(len(images)):
            if os.path.exists(os.path.join(video_dir,video_name,str(i)+".jpg")):
                predict_label,_ = model.predict1(os.path.join(video_dir,video_name,str(i)+".jpg"))
                result_file.write(str(i)+" #"+str(predict_label)+"\n")

        line = rf.readline()

def elder_main():
    '''
    model_dir = "/media/cql/DATA0/DEVELOPMENT/ai_4_eye_client_interface/temp_update/retina_quality.pth"
    test_db_quality("/media/cql/DATA0/DEVELOPMENT/xiangyaDB/","jpgs.txt",model_dir)
    
    model_name="densenet161"
    model_dir = "/data2/qilei_chen/DATA/GI_4_NEW/grayscale/"+model_name+"/best.model"
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
    videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
    videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
    videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    
    
    try:
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
    except:
        print("Error: unable to start thread")
    '''
    '''
    model_name="mobilenetv2"
    model_dir = "/data2/qilei_chen/DATA/5class_scene_alex/work_dir/mobilenetv2_2/best.model"
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
    videos_result_folder = os.path.join(videos_folder_dir,"5class_alex_manual_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
    videos_result_folder = os.path.join(videos_folder_dir,"5class_alex_manual_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    
    videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
    videos_result_folder = os.path.join(videos_folder_dir,"5class_alex_manual_"+model_name)
    process_4_situation_videos_gray(videos_folder_dir,model_dir,model_name,videos_result_folder)
    '''
    '''
    try:
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed2/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
        videos_folder_dir = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed_changjing20/"
        videos_result_folder = os.path.join(videos_folder_dir,"grayscale_"+model_name)
        threading.Thread( target=process_4_situation_videos_gray, args=(videos_folder_dir,model_dir,model_name,videos_result_folder) ).start()
    except:
        print("Error: unable to start thread")
    '''
    #test_videos()
    #create_confusion_matrix()
    #test_batch()
    #create_confusion_matrix_endoscope3()
    #test_on_videos_endoscope3()
    #create_confusion_matrix_huimangban()
    #test_on_videos_huimangban()
    pass
def sample_image_get_roi(video_dir,frame_index):

    cap = cv2.VideoCapture(video_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success,frame = cap.read()

    if success:
        save_dir = '/home/qilei/Downloads/changjing_issues/sample_frames/'
        crop_frame,roi = crop_img(frame)
        print(roi)
        cv2.imwrite(os.path.join(save_dir,os.path.basename(video_dir)+"_"+str(frame_index)+".jpg"),crop_frame)

def get_videos_rois():

    root_dir = '/home/qilei/Downloads/changjing_issues/'
    video_name = '20220127_112321_01_r04_olbs260.mp4'

    video_dir = os.path.join(root_dir,video_name)

    sample_image_get_roi(video_dir,16200)

    pass

def process_video_periods(model,video_dir,periods=[],roi = [668, 73, 1582, 925]):
    
    cap = cv2.VideoCapture(video_dir)
    result_file = open(video_dir+"_v2.txt",'w')
    for i in range(int(len(periods)/2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, periods[i*2])
        success, frame = cap.read()
        for j in range(periods[i*2],periods[i*2+1]):
            crop_frame = crop_img(frame,roi)
            label = model.predict(crop_frame)
            result_file.write(str(j)+" #"+str(label)+"\n")
            success, frame = cap.read()
            if not success:
                break

def process_model_on_videos():
    root_dir = '/home/qilei/Downloads/changjing_issues/'
    video_name = '20210408160002.avi'

    video_dir = os.path.join(root_dir,video_name)

    #pth_dir = '/home/qilei/.DEVELOPMENT/models/mobilenetv2_5class.pth'
    #model_name = 'mobilenetv2'
    #labels = [1,2,3,4,5]
    
    pth_dir = '/data/qilei/.DATASETS/Endoskop4INOUT/binary/work_dir/resnet152_2/best.model'
    model_name = 'resnet152'
    labels = [1,2]
    
    model = classifier(224,model_name=model_name,class_num_=len(labels))
    model.ini_model(pth_dir)
    process_video_periods(model,video_dir,[18000,18600],[300, 1, 1620, 1010])

if __name__ == "__main__":
    #get_videos_rois()
    process_model_on_videos()
    pass

