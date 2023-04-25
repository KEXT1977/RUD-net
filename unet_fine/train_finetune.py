import torch
import albumentations as A #图片处理的库
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim #实现了多种优化算法的包
from model_finetune import UNET
from utils_new import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_val_loader
)
import argparse
# from torch.utils.tensorboard import SummaryWriter
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time

class SummaryWriter:
    def init():
        None
    def add_scalar(arg,*args,**kwargs):
        None
    def close(*args,**kwargs):
        None
    def add_images(*args,**kwargs):
        None

T1 = time.time()

## Parameters Input
parser = argparse.ArgumentParser(description='input param')
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--output_img_path',  required=False,type=str, default='None')
parser.add_argument('--save_model_path',  required=False,type=str, default='Save/')
args = parser.parse_args()

if os.path.exists(args.save_model_path) == False:
    os.mkdir(args.save_model_path)

if args.output_img_path != 'None':
    if os.path.exists(args.output_img_path) == False:
        os.mkdir(args.output_img_path)

max_model = max([int(i.split('_')[-1].split('.')[0]) for i in os.listdir(args.model_path)])
    # max_model = 645
load_model_path = os.path.join(args.model_path,'my_checkpoint_'+str(max_model)+'.pth.tar')

## Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 400
NUM_WORKERS = 4
IMAGE_HEIGHT = 512  # 2048 originally
IMAGE_WIDTH = 512   # 2048 originally
PIN_MEMORY = True   #True会将数据放置到GPU上去（默认为False）
LOAD_MODEL = True  # 如果有权重设置为True
TRAIN_IMG_DIR =  "/data2/cxt/CT/CTDircadb1/origin/"
TRAIN_MASK_DIR = "/data2/cxt/CT/CTDircadb1/mask/"
VAL_IMG_DIR =    "/data2/cxt/CT/CTDircadb1/test/origin/"
VAL_MASK_DIR =   "/data2/cxt/CT/CTDircadb1/test/mask/"


NEW_VAL_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/train/origin"
NEW_MASK_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/train/mask"
TEST_VAL_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/test/origin"
TEST_MASK_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/test/mask"

FINE_TUNE = True

CHECK_ACC = True
MODEL_NUM = 390
log_name = 'U-Net-finetune_pred'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    model = model.cuda()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # 按照一维展开,一维处多个1
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
        # 自动混合精度，可以在神经网络推理过程中，针对不同的层，采用不同的数据精度进行计算，从而实现节省显存和加快速度的目的
        with torch.cuda.amp.autocast():
            
            predictions = model(data)
            
            # 衡量二者的差异
            
            loss = loss_fn(predictions, targets)
            
        # backward 每次batch训练更新一次
        optimizer.zero_grad()          # 将模型的参数梯度初始化为0
        scaler.scale(loss).backward()  # 反向传播计算梯度
        scaler.step(optimizer)         # 更新所有参数
        scaler.update()                # 看是否要增大scaler

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    

    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  #to函数，可以将tensor变量拷贝到gpu，多个gpu还可以写多线程。
    # 比nn.BCELoss () 多了一个sigmoid函数0 - 1
    # 预测结果pred的基础上先做了个sigmoid，然后继续正常算loss
    loss_fn = nn.BCEWithLogitsLoss()# binary cross entropy 二值交叉熵 
    # loss_fn = nn.CrossEntropyLoss()
    # 训练集全部 和 测试集全部
    train_loader, _ = get_loaders(
        NEW_VAL_IMG_DIR, 
        NEW_MASK_IMG_DIR,
        None,
        None,
        BATCH_SIZE,
        train_transforms,
        None,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    ## fine tune 
    writer = SummaryWriter()
    # class final_conv_new(nn.Module):
    #     def __init__(self,out_channels=1):
    #         super().__init__()  
    #         self.final_conv = nn.Sequential(
    #             nn.Conv2d(64,32, 3, 1, 1, bias=False),
    #             nn.Dropout(p=0.5),
    #             nn.BatchNorm2d(32),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(32,out_channels, kernel_size=1)
    #             )
    #     def forward(self,x):
    #         return self.final_conv(x)
    print("=> Loading checkpoint")
    model_paras = torch.load(load_model_path)
    model.load_state_dict(model_paras["state_dict"])
    for name, param in model.named_parameters():
        if 'down' in name :
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("=> Loading optimizer dict")
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(model_paras["optimizer"])
    
    if CHECK_ACC:
        print(12313)
        load_checkpoint(torch.load(os.path.join(args.save_model_path , 'my_checkpoint_{0}.pth.tar'.format(str(MODEL_NUM)))), model)
        print('ok')
        val_transforms = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],)
        val_loader = get_val_loader(TEST_VAL_IMG_DIR, TEST_MASK_IMG_DIR, val_transforms, NUM_WORKERS, PIN_MEMORY)
        check_accuracy(log_name,val_loader, model, device=DEVICE)
        # save_predictions_as_imgs(val_loader, model, folder=args.output_img_path,writer=writer, device=DEVICE)
        
    else:
        print(111111)

        # model.final_conv = final_conv_new()
        # model.final_conv = nn.Conv2d(64,1,1)

        # print(model)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            print("-------第{0}轮训练开始-------".format(epoch+1))
            train_fn(train_loader, model, optimizer, loss_fn, scaler)
            # 验证训练集
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data = data.to(device=DEVICE)
                    targets = targets.float().unsqueeze(1).to(device=DEVICE)
                    preds = model(data)
                    loss = loss_fn(preds,targets)
                    total_test_loss = total_test_loss + loss.item()
                print("整体测试集的loss: {0}".format(total_test_loss))
                with open('{0}_log_{1}_withtest_{2}.txt'.format(log_name,BATCH_SIZE,IMAGE_HEIGHT),'a') as f:
                    f.write('epoch:' + str(epoch+1)+ ' loss:' + str(total_test_loss) +'\n')
                writer.add_scalar('train_loss',total_test_loss,epoch)
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoches":epoch
            }

            if (epoch+1) % 15 ==0:
                save_checkpoint(checkpoint,os.path.join(args.save_model_path,'my_checkpoint_{0}.pth.tar'.format(epoch+1)))

        ## Notification 
        T2 = time.time()
        mail_host="smtp.qq.com"  #设置服务器
        mail_user="815991349"    #用户名
        mail_pass="xqoxmjoppninbefg"   #口令 
        sender = '815991349@qq.com'
        receivers = ['jdyx815991349@163.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
        content = '请到平台查收运行结果。\n本次运行时间为：{:.2f} s \n最终训练集的loss为：{:.4f}'.format(T2-T1,total_test_loss)
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header('AI Notification')
        message['To'] =  Header("Master Chen")
        subject = '模型训练结束'
        message['Subject'] = Header(subject, 'utf-8')
        try:
            smtpObj = smtplib.SMTP() 
            smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
            smtpObj.login(mail_user,mail_pass)  
            smtpObj.sendmail(sender, receivers, message.as_string())
            print ("邮件发送成功")
        except smtplib.SMTPException:
            print ("Error: 无法发送邮件")
        writer.close()

if __name__ == "__main__":
    main()
    # os.system('shutdown')