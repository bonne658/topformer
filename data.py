import os, sys, cv2, math
import numpy as np
import torch, glob
from torch.utils.data import Dataset

def RandomResizedCrop(im, mask):
	scales=(0.5, 2.)
	size=(448, 1088)
	crop_h, crop_w = size
	scale = np.random.uniform(min(scales), max(scales))
	im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
	im = cv2.resize(im, (im_w, im_h))
	mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
	if (im_h, im_w) == (crop_h, crop_w): return im, mask
	# pad to crop size if need
	pad_h, pad_w = 0, 0
	if im_h < crop_h: pad_h = (crop_h - im_h) // 2 + 1
	if im_w < crop_w: pad_w = (crop_w - im_w) // 2 + 1
	if pad_h > 0 or pad_w > 0:
		im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
		mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
	im_h, im_w, _ = im.shape
	sh, sw = np.random.random(2)
	# top left
	sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
	return im[sh:sh+crop_h, sw:sw+crop_w, :].copy(), mask[sh:sh+crop_h, sw:sw+crop_w].copy()

def adj_saturation(im, rate):
	M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
	shape = im.shape
	im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
	im = np.clip(im, 0, 255).astype(np.uint8)
	return im

def adj_brightness(im, rate):
	table = np.array([i * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
	return table[im]

def adj_contrast(im, rate):
	table = np.array([74 + (i - 74) * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
	return table[im]

def ColorJitter(im):
	a=[0.6, 1.4]
	rate = np.random.uniform(*a)
	adj_saturation(im, rate)
	rate = np.random.uniform(*a)
	adj_brightness(im, rate)
	rate = np.random.uniform(*a)
	adj_contrast(im, rate)
	return im

class LWDDataset(Dataset):
	def __init__(self, src_dir, train=True):
		self.jpgs = glob.glob(src_dir+'/*jpg')
		self.jpgs.sort()
		self.train = train
	def __len__(self):
		return len(self.jpgs)
	def __getitem__(self, idx):
		jpg = self.jpgs[idx]
		#print(jpg)
		png = jpg.replace('jpg', 'png')
		im = cv2.imread(jpg)
		if not self.train: 
			im = im.transpose(2, 0, 1).astype(np.float32)
			im /= 255.0
			return torch.from_numpy(im), jpg
		mask = cv2.imread(png, 0).astype(np.float32)
		mask /= 255.0
		mask = mask.astype(np.uint8)
		im, mask = RandomResizedCrop(im, mask)
		if np.random.random() < 0.5:
			im=im[:, ::-1, :].copy()
			mask = mask[:, ::-1].copy()
		im = ColorJitter(im)
		im = im.transpose(2, 0, 1).astype(np.float32)
		im /= 255.0
		return torch.from_numpy(im), torch.from_numpy(mask).long()
