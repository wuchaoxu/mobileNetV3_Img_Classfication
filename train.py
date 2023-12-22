import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_v3_large_model = mobilenet_v3_large(num_classes=10)


print("=========================================")


mobilenet_v3_large_model.to(device)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
testSet = torchvision.datasets.CIFAR10("./cifar10", train=False, transform=transform, download=True)
trainSet = torchvision.datasets.CIFAR10("./cifar10", train=True, transform=transform, download=True)


# #获取数据的相关信息
testSize = len(testSet)
trainSize = len(trainSet)
print(f"训练集长度为:{trainSize}")
print(f"测试集长度为:{testSize}")

testSetDataloader = DataLoader(testSet, batch_size=32, shuffle=True)
trainSetDataloader = DataLoader(trainSet, batch_size=32, shuffle=True)

#定义损失函数
lossFun = nn.CrossEntropyLoss()
lossFun = lossFun.to(device)  #调用gpu的第二种方法
lossFun.to(device)


'''选择优化器和学习率的调整方法'''
lr=0.01#定义学习率
optim=torch.optim.Adam(mobilenet_v3_large_model.parameters(),lr=lr)#导入网络和学习率
sculer=torch.optim.lr_scheduler.StepLR(optim,step_size=100)#调整优化器的学习率，步长设置为1
epoch = 50


for i  in range(epoch):
    print("==========第{}轮训练开始==========".format(i+1))

    total_train=0 #定义总损失

    mobilenet_v3_large_model.train()
    for data in trainSetDataloader:
        img,label=data
        with torch.no_grad():
            img =img.to(device)
            label=label.to(device)
        optim.zero_grad()
        output=mobilenet_v3_large_model.forward(img)  #前向传播
        train_loss = lossFun(output, label)
        train_loss.backward()#反向传播
        optim.step()#优化器更新
        total_train+=train_loss #损失相加


    sculer.step()  #调整学习率
    total_test=0#总损失
    total_accuracy=0#总精度


    mobilenet_v3_large_model.eval()
    for data in testSetDataloader:
        img,label =data #图片转数据
        with torch.no_grad():
            img=img.to(device)
            label=label.to(device)
            optim.zero_grad()#梯度清零
            out=mobilenet_v3_large_model(img)#投入网络
            test_loss=lossFun(out, label)
            total_test+=test_loss#测试损失，无反向传播
            accuracy=(out.argmax(1)==label).sum()#.clone().detach().cpu().numpy()#正确预测的总和比测试集的长度，即预测正确的精度
            total_accuracy+=accuracy
    print("第{}轮训练集上的损失：{}".format(i+1, total_train))
    print("第{}轮测试集上的损失：{}".format(i+1, total_test))
    print("测试集上的精度：{:.1%}".format(total_accuracy/testSize))#百分数精度，正确预测的总和比测试集的长度
    #{:.1%} 是百分数精度，可以将小数转换为百分数。其中，.1 表示保留一位小数。


    torch.save(mobilenet_v3_large_model.state_dict(),"mobileNetV3_{}.pth".format(i+1))


