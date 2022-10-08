import torch.nn as nn
import torchvision.models as models
from torch.nn.parameter import Parameter
import os
import pickle as pkl





model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2) ),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2) ),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2),  ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2),  ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    print("pretrained is ", pretrained)
    
    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = model.state_dict()

    if pretrained:
        #use alexnet pretrained weights
        for name, param in pret_net.items():
            print(name)
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue

        #xavier initialization of classification layers
        for layer in model.classifier:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)

        
            
    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #code borrowed from task2
    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = model.state_dict()
    if pretrained:
        #use alexnet pretrained weights
        for name, param in pret_net.items():
            print(name)
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue

        #xavier initialization of classification layers
        for layer in model.classifier:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)
    return model
