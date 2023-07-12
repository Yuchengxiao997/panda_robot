import torch
pthfile = r'policy.pth'
#net = torch.load(pthfile)
net.load_state_dict(torch.load(pthfile))
print(net)