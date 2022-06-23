import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET):label4
    dice2(TC):label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output,dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        target[target == 4] = 3
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target,self.n_classes)
        input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))