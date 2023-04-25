import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

torch.cuda.empty_cache()


# 定义一个双层卷积
# pytorch里面一切自定义操作基本上都是继承nn.Module类来实现的
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 调用父类的初始化函数
        super(DoubleConv,self).__init__()

        # 基础的序贯模型,将几个层包装在一起作为一个大的层（块）
        self.conv = nn.Sequential(
            # 因为后续有BatchNorm 会进行一个归一化类似的工作，所以不需要加偏置
            # in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=1
            # 即输 入/出 张量的channels数 即是颜色数 3--RGB 1--黑白
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # 三个通道  1  100 10000 --> 1 1.2  1.3 变为同个数量级
            # 是2016年的优化，添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_channels),
            # inplace会改变原输入
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()

        # 使用MoudleList的话，这个网络权重 (weithgs) 和偏置 (bias) 都会注册到U NET网络内。
        # 使用普通List能运行，但不能同步parameters
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 池化层：取出部分的最大值 kernel_size：用多大的框去取覆盖元素。
        # stride: 步长，是个tuple (1,2) 意味着上下步长为1，左右步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            # 最开始的时候加入的是 3-->64的
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        # up组成：一反+两卷
        for feature in reversed(features):
            self.ups.append(

        # 转置卷积.反卷积操作并不能还原出卷积之前的图片，只能还原出卷积之前图片的尺寸
        # out padding 是反卷积的时候为了防止因为/stride的时候向下取整而导致的不精确
        # 一般情况下的padding=(k-1)/2; o_p=s-1
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
        

    def forward(self, x):
        skip_connections = []
        # x=x.float()
        for down in self.downs:
            # down的时候每次都储存一下跳跃连接的输出。
            # 两次卷积+一次池化
            
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 底下中间的衔接部分
        x = self.bottleneck(x)

        # 倒转跳跃连接的部分
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # 0 2 4 6 ...
            # 一次反卷积+一次拼接+两次卷积
            # 反卷积
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                #print('resize,保险措施')

            # 在第一个维度进行拼接，不新增dim
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # 两次卷积
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3,3,512,512))# batch size 一次训练输入三张图片  1--通道数   w h
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()