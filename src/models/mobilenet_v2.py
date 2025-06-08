class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,stride,expand_ratio=1):
        super().__init__()
        self.stride=stride
        hidden_dim=int(in_channels*expand_ratio)# 增大（减小）通道数
        # 只有在输入与输出维度完全一致时才做跳连
        # stride=1时特征图尺寸不会改变；in_channels==out_channels，即输入输出通道数相同时，满足维度完全一致，因此可做跳连
        self.use_res_connect= self.stride==1 and in_channels==out_channels
        # 只有第一个bottleneck的expand_ratio=1(结构图中的t=1)，此时不需要前面的point wise conv
        if expand_ratio==1:
            self.conv=nn.Sequential(
                # depth wise conv
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=self.stride,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),# 由于expand_ratio=1，因此此时hideen_dim=in_channels
                nn.ReLU6(inplace=True),
                # point wise conv,线性激活（不加ReLU6）
                nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(out_channels)
                )
        # 剩余的bottlenek结构
        else:
            self.conv=nn.Sequential(
                # point wise conv
                nn.Conv2d(in_channels,hidden_dim,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depth wise conv
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=self.stride,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # point wise conv,线性激活（不加ReLU6）
                nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(out_channels)
                )
    def forward(self,x):
        if self.use_res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    #确保通道数减少量不能超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=1000,img_channel=3,width_mult=1.0):
        super().__init__()
        in_channels=32#第一个c
        last_channels=1280#最后的c
        #根据网络结构图得到如下网络配置
        inverted_residual_setting=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],            
        ]
        #1. building first layer,网络结构图中第一行的普通conv2d
        #这里的input_channel指的是第一个bottlenek的输入通道数
        input_channel = _make_divisible(in_channels * width_mult, 4 if width_mult == 0.1 else 8)
        #print(input_channel)#32
        layers=[self.conv_3x3_bn(in_channels=img_channel,out_channels=input_channel,stride=2)]
        #2. building inverted residual blocks
        for t,c,n,s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            #print(output_channel)#每次循环依次为：32,16,24,32,64,96,160,320
            for i in range(n):
                #InvertedResidual中的参数顺序：in_channels,out_channels,stride,expand_ratio
                layers.append(InvertedResidual(input_channel,output_channel,s if i==0 else 1,t))
                input_channel=output_channel#及时更新通道数
        self.features=nn.Sequential(*layers)
        #3. building last several layers
        output_channel = _make_divisible(last_channels * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else last_channels
        #print(output_channel)#1280
        #网络结构图中倒数第三行的普通conv2d
        self.conv=self.conv_1x1_bn(in_channels=input_channel,out_channels=output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
    def conv_3x3_bn(self,in_channels,out_channels,stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,groups=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    def conv_1x1_bn(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.conv(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x