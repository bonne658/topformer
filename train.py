import os, sys, cv2
import numpy as np
import torch, glob
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import LWDDataset
from topformer import TopFormer
from lr_scheduler import WarmupPolyLrScheduler
from CELoss import CELoss, OhemCELoss

bs = 4
# data
train_ds = LWDDataset("/home/hookii/lwd/data/lawn/train")
n_train = len(train_ds)/bs
train_loader = DataLoader(train_ds, shuffle=True, batch_size=bs, pin_memory=True, drop_last = True)
val_ds = LWDDataset("/home/hookii/lwd/data/lawn/val")
n_val = len(val_ds)/bs
val_loader = DataLoader(val_ds, shuffle=False, batch_size=bs, pin_memory=True, drop_last = False)
# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=TopFormer()
model.init_weights()
model.to(device=device)
# optim
lr_start=5e-3
optim = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=5e-4)
# lr scheduler
max_iter=15000
lr_schdr = WarmupPolyLrScheduler(optim, power=1.0, max_iter=max_iter, warmup_iter=100, warmup_ratio=0.1)
# loss
criteria_pre = CELoss()
criteria_pre = OhemCELoss(0.7)
#criteria_aux = [OhemCELoss(0.7) for _ in range(4)]

last = 1.5
vl = 11
for i in range(max_iter):
	model.train()
	train_loss = 0
	j = 0
	#print('lr:', lr_schdr.get_lr()[0])
	for it, (im, lb) in enumerate(train_loader):
		im = im.cuda()
		lb = lb.cuda()
		optim.zero_grad()
		logits = model(im)
		#logits = F.interpolate(logits, lb.shape[-2:], mode='bilinear', align_corners=True)
		loss_pre = criteria_pre(logits, lb)
		#loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
		loss = loss_pre #+ sum(loss_aux)
		loss.backward()
		optim.step()
		train_loss += loss
		print(i, j, loss.item(), 'lr:', lr_schdr.get_lr()[0], 'val_loss:', vl)
		j+=1
	print('train_loss:', train_loss.item()/n_train)
	lr_schdr.step()
	
	model.eval()
	val_loss = 0
	with torch.no_grad():
		for it, (im, lb) in enumerate(val_loader):
			im = im.cuda()
			lb = lb.cuda()
			logits = model(im)
			#logits = F.interpolate(logits, lb.shape[-2:], mode='bilinear', align_corners=True)
			loss_pre = criteria_pre(logits, lb)
			val_loss += loss_pre
		vl = val_loss.item()/n_val
		#print('val_loss:', vl)
	print('*'*33)
	if vl < last:
		last = vl
		torch.save(model.state_dict(), 'result/model/'+str(i)+'--'+str(vl)[:7]+'.pth')
