import torch
import torchvision
from dataset import VesselDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# 保存权重，网络参数
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    # 仅仅载入了model的参数
    model.load_state_dict(checkpoint["state_dict"])

# 返回两个可迭代对象，分别是训练集和测试集的(样本,标签）
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=False,
):
    train_ds = VesselDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    # dataloader的dataset参数接受的是Dataset对象！
    # dataloader的对象是可迭代输出的
    train_loader = DataLoader(
        # 训练集数据（包括样本和标签的元组大集合）
        train_ds,
        # 从数据库中每次抽出batch size个样本
        batch_size=batch_size,
        # 多少个子线程用于加载数据
        num_workers=num_workers,
        # 锁页内存，True则是锁页，不会与硬盘进行交换，只在gpu中运行，默认选False
        pin_memory=pin_memory,
        shuffle=True,
    )

    # val_ds = VesselDataset(
    #     image_dir=val_dir,
    #     mask_dir=val_maskdir,
    #     transform=val_transform,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    return train_loader,None

def get_val_loader(val_dir,val_maskdir,val_transform,num_workers=4,pin_memory=False,names=True):

    val_ds = VesselDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        names = names
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return val_loader

def check_accuracy(log_name,loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    dice_score_all = 0
    count = 0
    correct_pixels = 0
    total_pixels = 0
    
    model.eval()
    with torch.no_grad():
        for x, y, z in tqdm(loader):
            x = x.to(device)

            # 将验证集的标签按照一维展开。
            y = y.to(device).unsqueeze(1)
            
            # 映射到0-1之间
            preds = torch.sigmoid(model(x))

            # print('y的shape：',y.shape)
            # print('preds的shape：',preds.shape)

            # 把预测结果里的大于0.5的全部变成1，其余变成0
            preds = (preds > 0.5).float()
            # preds 和 y 之间一样的像素
            num_correct += 2*(preds * y).sum()
            
            # preds这个tensor里一共有多少个元素
            num_pixels += (preds + y).sum()
            dice_score = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # Pixel Accuracy
            # 比较预测结果和真实值，得到一个布尔值tensor
            correct = torch.eq(preds, y)

            # 计算所有像素点的总数
            total_pixels += torch.numel(y)

            # 计算预测正确的像素数量
            correct_pixels += torch.sum(correct)
            #print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
            #print('loader的长度：',len(loader))
            dice_score_all +=dice_score
            count += 1
            with open('{0}_dice_score.txt'.format(log_name),'a') as f:
                f.write(f"{z[0]} Dice score: {dice_score}"+'\n')
            
        pixel_acc = correct_pixels.item() / total_pixels
        with open('{0}_dice_score.txt'.format(log_name),'a') as f:
            f.write("Mean Dice Score: {}".format(dice_score_all/count)+'\n')
            f.write("All Dice Score: {}".format(num_correct/num_pixels)+'\n')
            f.write("Pixel Accuracy: {}".format(pixel_acc)+'\n')
        
        print("Mean Dice Score: {}".format(dice_score_all/count))
        print("All Dice Score: {}".format(num_correct/num_pixels))
        print("Pixel Accuracy: {}".format(pixel_acc))
        
def save_predictions_as_imgs(loader, model, folder="saved_images/",writer=None, device="cuda"):

    # eval() :不启用 BatchNormalization 和 Dropout.保证BN和dropout不发生变化,用训练好的值
    # 否则一旦test的batch_size过小，很容易就会被BN层影响结果
    model.eval()
    try:
        for idx, (x, y, z) in enumerate(tqdm(loader)):

            x = x.to(device=device)
            y = y.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, os.path.join(folder,"{0}".format(z[0])))

            writer.add_images(z[0]+' and Ground Truth',torch.cat([y.unsqueeze(1),preds]),idx)
        writer.close()
    except:
        for idx, (x, z) in enumerate(tqdm(loader)):

            x = x.to(device=device)
            
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, os.path.join(folder,"{0}_pred.png".format(z[0].replace('.png',''))))