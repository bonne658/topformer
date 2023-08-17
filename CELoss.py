import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.criteria = nn.CrossEntropyLoss(reduction='mean')
	def forward(self, logits, labels):
		# bchw
		ls = F.log_softmax(logits, dim=1).contiguous()
		# for ignore
		mask = F.one_hot(labels, 8).float()
		# bhwc
		mask = mask[...,:logits.shape[1]]
		# bchw
		mask = mask.permute(0, 3, 1, 2).contiguous()
		loss = -mask * ls
		# bhw
		loss = loss.sum(dim=1)
		return torch.mean(loss)

class OhemCELoss(nn.Module):

    def __init__(self, thresh, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.lb_ignore = lb_ignore

    def forward(self, logits, labels):
        #print(labels.shape)
        n_min = labels[labels != self.lb_ignore].numel() // 16
        #loss = self.criteria(logits, labels).view(-1)
        ls = F.log_softmax(logits, dim=1).contiguous()
        #print(ls)
        mask = F.one_hot(labels, logits.shape[1]).float()
        mask = mask.permute(0, 3, 1, 2).contiguous()
        #print(mask.shape, ls.shape)
        #print(n_min)
        loss = -mask * ls
        loss = loss.sum(dim=1)
        
        loss = loss.contiguous().view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)



