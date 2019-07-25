import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ================================================================================
# region focalloss_creating NaN
# class FocalLoss(nn.Module):
#     """
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#         Focal_Loss= -1*alpha*(1-pt)*log(pt)
#     :param num_class:
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                     focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """

#     def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.num_class = num_class
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.size_average = size_average

#         if self.alpha is None:
#             self.alpha = torch.ones(self.num_class, 1)
#         elif isinstance(self.alpha, (list, np.ndarray)):
#             assert len(self.alpha) == self.num_class
#             self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
#             self.alpha = self.alpha / self.alpha.sum()
#         elif isinstance(self.alpha, float):
#             alpha = torch.ones(self.num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[balance_index] = self.alpha
#             self.alpha = alpha
#         else:
#             raise TypeError('Not support alpha type')

#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')

#     def forward(self, logit, target):

#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = target.view(-1, 1)

#         epsilon = 1e-10
#         alpha = self.alpha
#         if alpha.device != logit.device:
#             alpha = alpha.to(logit.device)

#         idx = target.cpu().long()

#         one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)

#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + epsilon
#         logpt = pt.log()

#         gamma = self.gamma

#         alpha = alpha[idx]
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
# endregion 

# region 
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
# endregion 

# region 
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
# endregion 

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

def focal_loss(inputs, targets, alpha=1, gamma=2):
  # print("targets",targets)
  # tensor([[0., 0., 0., 0., 1., 0., 0.],
  #         [0., 0., 1., 0., 0., 0., 0.],
  #         [0., 0., 0., 0., 0., 0., 1.],
  #         [0., 0., 0., 0., 0., 0., 1.],
  #         [0., 0., 0., 1., 0., 0., 0.],
  #         [0., 0., 0., 0., 0., 1., 0.],
  
  BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
  pt = torch.exp(-BCE_loss) # prevents nans when probability 0
  F_loss = alpha * (1-pt)**gamma * BCE_loss
  return F_loss.mean()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loss_visualization(train_loss_record,validate_loss_record,epoch_num):
  fig=plt.figure()

  # ================================================================================
  axes1=plt.subplot(1,2,1)
  axes1.plot(train_loss_record)

  # plt.xlim(45,85)
  # plt.title("Train loss values (focal loss) / epochs: "+str(epoch_num))
  plt.title("Train loss values / epochs: "+str(epoch_num))
  plt.xlabel("Time")
  plt.ylabel("Loss value")

  # ================================================================================
  axes2=plt.subplot(1,2,2)
  axes2.plot(validate_loss_record)

  # plt.xlim(45,85)
  plt.title("Validation loss values / epochs: "+str(epoch_num))
  plt.xlabel("Time")
  plt.ylabel("Loss value")

  # ================================================================================
  plt.subplots_adjust(wspace=None,hspace=0.3)
  plt.tight_layout()
  plt.show()

