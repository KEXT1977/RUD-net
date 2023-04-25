import torch
import albumentations as A #图片处理的库
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim #实现了多种优化算法的包
from model import UNET
from utils import (
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
    def __init__(arg,*args,**kwargs):
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
args = parser.parse_args()

if os.path.exists(args.model_path) == False:
    os.mkdir(args.model_path)

load_model = False
if args.output_img_path != 'None':
    if os.path.exists(args.output_img_path) == False:
        os.mkdir(args.output_img_path)
    load_model = True
    max_model = max([int(i.split('_')[-1].split('.')[0]) for i in os.listdir(args.model_path)])
    # max_model = 645
    load_model_path = os.path.join(args.model_path,'my_checkpoint_'+str(max_model)+'.pth.tar')

## Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 400
NUM_WORKERS = 4
IMAGE_HEIGHT = 512  # 2048 originally
IMAGE_WIDTH = 512   # 2048 originally
PIN_MEMORY = True   #True会将数据放置到GPU上去（默认为False）
LOAD_MODEL = load_model  # 如果有权重设置为True
TRAIN_IMG_DIR =  "/data2/cxt/CT/CTDircadb1/origin/"
TRAIN_MASK_DIR = "/data2/cxt/CT/CTDircadb1/mask/"
VAL_IMG_DIR =    "/data2/cxt/CT/CTDircadb1/test/origin/"
VAL_MASK_DIR =   "/data2/cxt/CT/CTDircadb1/test/mask/"

NEW_DATA = True
NEW_VAL_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/test/origin"
NEW_MASK_IMG_DIR = "/data2/cxt/CT/hospital_data_clean/png/finetune/test/mask"

log_name = 'U-Net_pred'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练集全部 和 测试集全部
    train_loader, _ = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    writer = SummaryWriter()
    if LOAD_MODEL:

        load_checkpoint(torch.load(load_model_path), model)# 当然如果设置的权重名称或路径不是这个，要修改的
        # 重新导入新的数据集！
        if not NEW_DATA:
            val_loader = get_val_loader(VAL_IMG_DIR, VAL_MASK_DIR, val_transforms, NUM_WORKERS, PIN_MEMORY)
            check_accuracy(val_loader, model, device=DEVICE)
            save_predictions_as_imgs(val_loader, model, folder=args.output_img_path,writer=writer, device=DEVICE)
        else:
            val_loader = get_val_loader(NEW_VAL_IMG_DIR, NEW_MASK_IMG_DIR, val_transforms, NUM_WORKERS, PIN_MEMORY)
            check_accuracy(log_name,val_loader, model, device=DEVICE)
            save_predictions_as_imgs(val_loader, model, folder=args.output_img_path,writer=writer, device=DEVICE)

                
    else:
            
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
                    f.write('epoch:' + str(epoch)+ ' loss:' + str(total_test_loss) +'\n')
                    writer.add_scalar('train_loss',total_test_loss,epoch)

                    # save model
            
            if (epoch+1) % 15 == 0:
                checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "epoches":epoch
                                        }
                save_checkpoint(checkpoint,os.path.join(args.model_path,'my_checkpoint_{0}.pth.tar'.format(epoch+1)))

        ## Notification 

        T2 = time.time()

        mail_host="smtp.qq.com"  #设置服务器
        mail_user=""    #用户名
        mail_pass=""   #口令 

        sender = ''
        receivers = ['']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
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