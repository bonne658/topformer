import torch, sys
from collections import OrderedDict
from topformer import TopFormer

net=TopFormer()
paras = torch.load('result/model/637--0.00634.pth', map_location="cpu")
net.eval()
dct = OrderedDict()
for key in paras:
	if not 'aux' in key: dct[key]=paras[key]
#sys.exit()
net.load_state_dict(dct)
x = torch.randn((1, 3, 512, 1280))
traced_script_module = torch.jit.trace(net, x)
traced_script_module.save('637.pt')
