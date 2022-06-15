import torch.nn as nn
import torch
import numpy as np
from models.spectral_normalization import SpectralNorm

from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor 
import torch.autograd as autograd
import scipy.io
import argparse, json, glob, os

from models.unet import Unet

def t32(x):
    return torch.transpose(x,0, 2).squeeze(2)
def t23(x):
    return torch.transpose(x, 0,1).unsqueeze(0)

def t32_1(x):
    x= torch.transpose(x,1, 2)
    return x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
def t23_1(x):
    return x.reshape(-1,16,x.shape[1],x.shape[2],x.shape[3]).transpose(1, 2)

def tv_loss(x, beta = 0.5):
    dh = torch.pow(x[...,1:] - x[...,:-1], 2)
    dw = torch.pow(x[...,1:,:] - x[...,:-1,:], 2)
    dt = torch.pow(x[...,1:,:,:] - x[...,:-1,:,:], 2)
    
    return torch.sum(dh[..., :-1, :-1,:] + dw[..., :-1, :, :-1] + dt[...,:-1,:-1] )

def remove_nans(x):
    for k in x.parameters():
        if torch.any(torch.isnan(k.grad)):
            k.grad[torch.isnan(k.grad)] = torch.zeros(1, device = k.device)

def load_generator(folder_name, device):
    
    device = 'cuda:'+str(device)
    parser = argparse.ArgumentParser(description='Process some integers.')
    args1 = parser.parse_args('')
    with open(folder_name + 'args.txt', 'r') as f:
        args1.__dict__ = json.load(f)

    list_of_files = glob.glob(folder_name + 'bestgen*.pt') 
    chkp_path = max(list_of_files, key=os.path.getctime)

    if args1.network == 'noUnet':
        model1 = None 
    else:
        if args1.network == 'Unet_cat':
            in_channels = 8
        else:
            in_channels = 4

        
        res_opt = bool(args1.unet_opts.split('_')[0].split('residual')[-1]) 
        model1 = Unet(n_channel_in=in_channels, 
                     n_channel_out=4, 
                     residual=res_opt, 
                    down=args1.unet_opts.split('_')[1], 
                     up=args1.unet_opts.split('_')[2], 
                     activation=args1.unet_opts.split('_')[3])#.to(device)
    
    generator = NoiseGenerator2d_withFixed(net = model1, unet_opts = args1.network, device = device)#.to(device)

    saved_state_dict = torch.load(chkp_path, map_location=device)
    generator.load_state_dict(saved_state_dict)
    #generator.load_state_dict(torch.load(chkp_path, map_location=device))
    return generator

def load_generator2(folder_name, device):
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args('')
    with open(folder_name + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
        args.fraction_video = 50
        args.resume_from_checkpoint = folder_name
        

    if args.network == 'noUnet':
        model = None 
    else:
        if args.network == 'Unet_cat':
            in_channels = 8
        else:
            in_channels = 4

        res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1]) 
        model = Unet(n_channel_in=in_channels, 
                     n_channel_out=4, 
                     residual=res_opt, 
                    down=args.unet_opts.split('_')[1], 
                     up=args.unet_opts.split('_')[2], 
                     activation=args.unet_opts.split('_')[3])#.to(args.device)

    generator = NoiseGenerator2d3d_distribubted(net = model, unet_opts = args.network, add_fixed = args.addfixed, 
                                               device = device)
    
    
    #list_of_files = glob.glob(folder_name + '/bestgen*.pt') # * means all if need specific format then *.csv
    list_of_files = glob.glob(folder_name + '/gen*.pt') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location='cuda:'+str(device))
    distributed_model = False
    for key in saved_state_dict:
        if 'module' in key:
            distributed_model = True
            print('distributed')
            break
        
    if distributed_model == True:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        saved_state_dict = new_state_dict
        
    generator.load_state_dict(saved_state_dict)
    
    #generator.load_state_dict(torch.load(path, map_location='cuda:'+str(device)))
    
    #curr_epoch = int(path.split('/')[-1].split('_')[0].split('bestgenerator')[1])
    curr_epoch = int(path.split('/')[-1].split('_')[0].split('generatorcheckpoint')[1])
    print('current epoch', curr_epoch)
    
    if args.network != 'noUnet':
        generator.net.conv1.dropout = False
        generator.net.conv2.dropout = False
        generator.net.conv3.dropout = False
        generator.net.conv4.dropout = False
        generator.net.conv5.dropout = False
        generator.net.conv6.dropout = False
        generator.net.conv7.dropout = False
        generator.net.conv8.dropout = False
        generator.net.conv9.dropout = False
        generator.net.convres.dropout = False

    return generator

def load_from_checkpoint_ab(folder_name, device='cuda:0', ep='latest', new_model = False):
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args('')
    with open(folder_name + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
        args.fraction_video = 50
        args.resume_from_checkpoint = folder_name
        
    if new_model == True:
        import models.fastdvdnet as fdvd
        model = fdvd.DenBlockUnet(num_input_frames=1)#.to(args.device)
    else:
        if args.network == 'noUnet':
            model = None 
        else:
            if args.network == 'Unet_cat':
                in_channels = 8
            else:
                in_channels = 4

            res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1]) 
            model = Unet(n_channel_in=in_channels, 
                         n_channel_out=4, 
                         residual=res_opt, 
                        down=args.unet_opts.split('_')[1], 
                         up=args.unet_opts.split('_')[2], 
                         activation=args.unet_opts.split('_')[3])#.to(args.device)

    generator = NoiseGenerator2d3d_distributed_ablation(net = model, unet_opts = args.network, noise_list = args.noiselist, 
                                               device = device)
    
    if ep == 'best':
        list_of_files = glob.glob(folder_name + '/bestgen*.pt') # * means all if need specific format then *.csv
        kld_best = []
        for i in range(0,len(list_of_files)):
            kld_best.append(float(list_of_files[i].split('KLD')[-1].split('.pt')[0]))
            
        inds_sorted = np.argsort(kld_best)
        best_files = np.array(list_of_files)[inds_sorted]

        latest_file = best_files[0]
        
        print('best kld:' , np.min(kld_best))
        #list_of_files = glob.glob(folder_name + '/bestgen*.pt') # * means all if need specific format then *.csv
        #latest_file = max(list_of_files, key=os.path.getctime)
    elif ep == 'latest':
        list_of_files = glob.glob(folder_name + '/gen*.pt') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        list_of_files = glob.glob(folder_name + '/generatorcheckpoint' + str(ep) +'_' + '*.pt')
        #print(list_of_files)
        latest_file = list_of_files[0]
        
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location ='cuda:'+str(device))
    distributed_model = False
    for key in saved_state_dict:
        if 'module' in key:
            distributed_model = True
            print('distributed')
            break
        
    if distributed_model == True:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        saved_state_dict = new_state_dict
        
    generator.load_state_dict(saved_state_dict)
    
    if ep == 'best':
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('bestgenerator')[1])
    else:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('generatorcheckpoint')[1])
    print('current epoch', curr_epoch)

    return generator
            
class KLD_loss(nn.Module):
    def __init__(self, nbins = 1000, xrange = (0,1)):
        super(KLD_loss, self).__init__()
        self.nbins = nbins
        self.xrange = xrange

    def forward(self, x1, x2):
        sz = np.prod(list(x1.shape))
        p = torch.histc(x1, bins = self.nbins, min =self.xrange[0], max = self.xrange[1])/sz
        q = torch.histc(x2, bins = self.nbins, min =self.xrange[0], max = self.xrange[1])/sz
        idx = (p > 0) & (q > 0)
        p = p[idx]
        q = q[idx]
        logp = torch.log(p)
        logq = torch.log(q)
        kl_fwd = torch.sum(p * (logp - logq))
        kl_inv = torch.sum(q * (logq - logp))
        kl_sym = (kl_fwd + kl_inv) / 2.0

        return kl_sym
    
def split_into_patches(x, patch_size = 64):
    patches = torch.empty([1,4,16,patch_size,patch_size])
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

def split_into_patches2d(x, patch_size = 64):
    patches = torch.empty([1,x.shape[1],patch_size,patch_size], device = x.device)
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers

def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    bin_edges = None
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_sym #kl_fwd #, kl_inv, kl_sym

def compute_gradient_penalty2d(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype = real_samples.dtype, device = real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[...,0]
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)
    
    #print(d_interpolates.shape, interpolates.shape, fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1,1)), dtype = real_samples.dtype, device = real_samples.device)
    # Get random interpolation between real and fake samples
    #print(alpha.shape, fake_samples.shape)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)
    
    #print(d_interpolates.shape, interpolates.shape, fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class NoiseGenerator2d_withFixed(nn.Module):
    def __init__(self, net, unet_opts = 'Unet', device = 'cuda:0'):
        super(NoiseGenerator2d_withFixed, self).__init__()
        
        print(device)
        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        self.net = net
        
        self.unet_opts = unet_opts
        
        mean_noise = scipy.io.loadmat('../data/fixed_pattern_noise.mat')['mean_pattern']
        fixed_noise = mean_noise.astype('float32')/2**16
        self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1).copy(), dtype = torch.float32, device = device).unsqueeze(0)

        print(device)
    def forward(self, x, i0=None):
        
        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        if self.fixednoiset.shape[-2] == x.shape[-2]:
            i1 = 0
        else:
            i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
        if self.fixednoiset.shape[-1] == x.shape[-1]:
            i2 = 0
        else:
            i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            
        if i0 is not None:
            print('here I am right now!!!!')
            i1 = i0[0]
            i2 = i0[1]
            
        fixed_noise =  self.fixednoiset[...,i1:i1+x.shape[-2], i2:i2 + x.shape[-1]]
                
        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)

        #print(shot_noise.shape, fixed_noise.shape)
        noise = shot_noise + uniform_noise + row_noise + fixed_noise
        
        noisy = x + noise
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy
class NoiseGenerator2d(nn.Module):
    def __init__(self, net, unet_opts = 'Unet', device = 'cuda:0'):
        super(NoiseGenerator2d, self).__init__()
        
        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        self.net = net
        
        self.unet_opts = unet_opts
        
    def forward(self, x):
        
        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)

        noise = shot_noise + uniform_noise + row_noise
        
        noisy = x + noise
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy
class NoiseGenerator2d3d_distribubted(nn.Module):
    def __init__(self, net, unet_opts = 'Unet', device = 'cuda:0', add_fixed = 'True'):
        super(NoiseGenerator2d3d_distribubted, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        self.net = net
        
        self.unet_opts = unet_opts
        
        mean_noise = scipy.io.loadmat('../data/fixed_pattern_noise.mat')['mean_pattern']
        fixed_noise = mean_noise.astype('float32')/2**16
        if 'learned' in add_fixed:
            print('using learned fixed noise')
            fixed_init = torch.zeros((1,fixed_noise.shape[-1],1,fixed_noise.shape[1]), dtype = self.dtype, device = device)
            self.fixednoiset = torch.nn.Parameter(fixed_init, requires_grad = True)
        else:
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
        
        
        if 'periodic' in add_fixed:
            self.periodic_params = torch.nn.Parameter(torch.tensor([0.0050,0.0050,0.0050], dtype = self.dtype, device = device), requires_grad = True)
        
        self.add_fixed  = add_fixed 
        self.indices = None
        self.keep_track = False
        self.all_noise = {}
        
    def forward(self, x, split_into_patches = False, i0=None):
        
        #print(x.shape)
        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        if 'True' in self.add_fixed:
            #i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
            #i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            if self.indices is not None:
                i1 = self.indices[0]
                i2 = self.indices[1]
            elif i0 is not None:
                i1 = i0[0]
                i2 = i0[1]
            else:
                
                i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
                i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            fixed_noise = self.fixednoiset[...,i1:i1+x.shape[-2], i2:i2 + x.shape[-1]]
            #if self.keep_track == True:

            

        #print(x.shape, fixed_noise.shape)
        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
        row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)

        
        if self.keep_track == True:
            self.all_noise['shot_read'] = shot_noise.detach().cpu().numpy() 
            self.all_noise['unif'] = uniform_noise.detach().cpu().numpy() 
            self.all_noise['row'] = np.repeat(row_noise.detach().cpu().numpy(), self.all_noise['shot_read'].shape[2], axis=2)
            self.all_noise['fixed'] = fixed_noise.detach().cpu().numpy()
        
        if 'True' in self.add_fixed:
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp + fixed_noise
        elif 'learned' in self.add_fixed:
            fixed_noise = self.fixednoiset[...,self.indices[2]:self.indices[3]]
            #print('here', shot_noise.shape, fixed_noise.shape)
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp + fixed_noise
        else: 
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp
            
        if 'periodic' in self.add_fixed:
            periodic_noise = torch.zeros(x.shape,  dtype=torch.cfloat, device = self.device)
            
            periodic_noise[...,0,0] = self.periodic_params[0]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            
            periodic0 = self.periodic_params[1]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            periodic1 = self.periodic_params[2]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device) 
            periodic_noise[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            periodic_noise[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)

            periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise, norm="ortho"))

            noise = noise + periodic_gen
            if self.keep_track == True:
                self.all_noise['periodic'] = periodic_gen.detach().cpu().numpy() 
            
        noisy = x + noise
        
        if split_into_patches== True:
            noisy = split_into_patches2d(noisy)
            x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy    
    
    
class NoiseGenerator2d3d_distributed_ablation(nn.Module):
    def __init__(self, net, unet_opts = 'noUnet', device = 'cuda:0', noise_list = 'shot_read_row'):
        super(NoiseGenerator2d3d_distributed_ablation, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.noise_list = noise_list
        self.net = net
        self.unet_opts = unet_opts
        self.keep_track = False
        self.all_noise = {}
        
        if 'shot' in noise_list:
            self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002*10000, dtype = self.dtype, device = device), 
                                                 requires_grad = True)
        if 'read' in noise_list:     
            self.read_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device), 
                                                 requires_grad = True)
        if 'row1' in noise_list:         
            self.row_noise = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), 
                                                requires_grad = True)
        if 'rowt' in noise_list:
            self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), 
                                                     requires_grad = True)
        if 'uniform' in noise_list:    
            self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001*10000, dtype = self.dtype, device = device), requires_grad = True)
        if 'fixed1' in noise_list:
            mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
        if 'learnedfixed' in noise_list:
            print('using learned fixed noise')
            
            mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
            self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)
        
        if 'periodic' in noise_list:
            self.periodic_params = torch.nn.Parameter(torch.tensor([0.0050,0.0050,0.0050], 
                                                                   dtype = self.dtype, device = device)*100, #*1000, 
                                                                  requires_grad = True)
        
        self.indices = None
        
        
    def forward(self, x, split_into_patches = False, i0=None):

        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        noise = torch.zeros_like(x)
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            variance = x*self.shot_noise + self.read_noise
            shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise += shot_noise
            if self.keep_track == True:
                self.all_noise['shot_read'] = shot_noise.detach().cpu().numpy() 
        elif 'read' in self.noise_list:
            variance =self.read_noise
            noise += torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        if 'uniform' in self.noise_list:    
            uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
            noise += uniform_noise
            if self.keep_track == True:
                self.all_noise['uniform'] = uniform_noise.detach().cpu().numpy() 
        if 'row1' in self.noise_list: 
            row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
            noise += row_noise
            if self.keep_track == True:
                self.all_noise['row'] = row_noise.detach().cpu().numpy() 
        if 'rowt' in self.noise_list:   
            row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)
            noise += row_noise_temp
            if self.keep_track == True:
                self.all_noise['rowt'] = row_noise_temp.detach().cpu().numpy() 
        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            if self.indices is not None:
                i1 = self.indices[0]
                i2 = self.indices[1]
            elif i0 is not None:
                i1 = i0[0]
                i2 = i0[1]
            else:
                i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
                i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            fixed_noise = self.fixednoiset[...,i1:i1+x.shape[-2], i2:i2 + x.shape[-1]]
            
            noise += fixed_noise
            if self.keep_track == True:
                self.all_noise['fixed'] = fixed_noise.detach().cpu().numpy() 
            
        #elif 'learnedfixed' in self.noise_list:
        #    fixed_noise = self.fixednoiset[...,self.indices[2]:self.indices[3]]
        #    noise += fixed_noise

        if 'periodic' in self.noise_list:
            periodic_noise = torch.zeros(x.shape,  dtype=torch.cfloat, device = self.device)
            periodic_noise[...,0,0] = self.periodic_params[0]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            
            periodic0 = self.periodic_params[1]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            periodic1 = self.periodic_params[2]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device) 

            periodic_noise[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            periodic_noise[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)

            periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise, norm="ortho"))

            noise += periodic_gen
            if self.keep_track == True:
                self.all_noise['periodic'] = periodic_gen.detach().cpu().numpy() 
    
            
        noisy = x + noise
        
        if split_into_patches== True:
            noisy = split_into_patches2d(noisy)
            x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy    
class NoiseGenerator2d3d(nn.Module):
    def __init__(self, net, unet_opts = 'Unet', device = 'cuda:0', add_fixed = 'True'):
        super(NoiseGenerator2d3d, self).__init__()
        
        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        self.net = net
        
        self.unet_opts = unet_opts
        
        mean_noise = scipy.io.loadmat('../data/fixed_pattern_noise.mat')['mean_pattern']
        fixed_noise = mean_noise.astype('float32')/2**16
        if 'learned' in add_fixed:
            print('using learned fixed noise')
            fixed_init = torch.zeros((1,fixed_noise.shape[-1],1,fixed_noise.shape[1]), dtype = self.dtype, device = device)
            self.fixednoiset = torch.nn.Parameter(fixed_init, requires_grad = True)
        else:
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
        
        
        if 'periodic' in add_fixed:
            self.periodic_params = torch.nn.Parameter(torch.tensor([0.0050,0.0050,0.0050], dtype = self.dtype, device = device), requires_grad = True)
        
        self.add_fixed  = add_fixed 
        self.indices = None
        
    def forward(self, x, split_into_patches = False, i0=None):
        
        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        if 'True' in self.add_fixed:
            #i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
            #i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            if self.indices is not None:
                i1 = self.indices[0]
                i2 = self.indices[2]
            elif i0 is not None:
                i1 = i0[0]
                i2 = i0[1]
            else:
                i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
                i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            fixed_noise = self.fixednoiset[...,i1:i1+x.shape[-2], i2:i2 + x.shape[-1]]
            

        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
        row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)

        if 'True' in self.add_fixed:
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp + fixed_noise
        elif 'learned' in self.add_fixed:
            fixed_noise = self.fixednoiset[...,self.indices[2]:self.indices[3]]
            #print('here', shot_noise.shape, fixed_noise.shape)
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp + fixed_noise
        else: 
            noise = shot_noise + uniform_noise + row_noise + row_noise_temp
            
        if 'periodic' in self.add_fixed:
            periodic_noise = torch.zeros(x.shape,  dtype=torch.cfloat, device = self.device)
            periodic_noise[...,x.shape[-2]//2,0] = self.periodic_params[0]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            periodic0 = self.periodic_params[1]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            periodic1 = self.periodic_params[2]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device) 
            periodic_noise[...,x.shape[-2]//2,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            periodic_noise[...,x.shape[-2]//2,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)

            periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise, norm="ortho"))

            noise = noise + periodic_gen
    
            
        noisy = x + noise
        
        if split_into_patches== True:
            noisy = split_into_patches2d(noisy)
            x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy
    
class NoiseGenerator(nn.Module):
    def __init__(self, net, device = 'cuda:0'):
        super(NoiseGenerator, self).__init__()

        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        self.net = net
        
    def forward(self, x):
        
        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)

        row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)

        noise = shot_noise + uniform_noise + row_noise + row_noise_temp
        
        noisy = x + noise
        noisy  = self.net(noisy)
        noisy = torch.clip(noisy, 0, 1)

        return noisy
    

class NoiseGenerator_nounet(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(NoiseGenerator_nounet, self).__init__()

        self.device = device
        self.dtype = torch.float32
        self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002, dtype = self.dtype, device = device), requires_grad = True)
        self.read_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002, dtype = self.dtype, device = device), requires_grad = True)
        self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001, dtype = self.dtype, device = device), requires_grad = True)
        
    def forward(self, x):
        
        variance = x*self.shot_noise + self.read_noise
        shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
        row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)

        row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)

        noise = shot_noise + uniform_noise + row_noise + row_noise_temp
        
        noisy = x + noise
        noisy = torch.clip(noisy, 0, 1)

        return noisy

channels = 4
leak = 0.1
w_g = 8

class DiscriminatorS(nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv3d(channels, 64, 3, stride=1, padding=(1,1,1)))

        self.conv2 = SpectralNorm(nn.Conv3d(64, 64, 4, stride=2, padding=(1,1,1)))
        self.conv3 = SpectralNorm(nn.Conv3d(64, 128, 3, stride=1, padding=(1,1,1)))
        self.conv4 = SpectralNorm(nn.Conv3d(128, 128, 4, stride=2, padding=(1,1,1)))
        self.conv5 = SpectralNorm(nn.Conv3d(128, 256, 3, stride=1, padding=(1,1,1)))
        self.conv6 = SpectralNorm(nn.Conv3d(256, 256, 4, stride=2, padding=(1,1,1)))
        self.conv7 = SpectralNorm(nn.Conv3d(256, 512, 3, stride=1, padding=(1,1,1)))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        return self.fc(m.view(-1,w_g * w_g * 512)).view(-1)
    
channels = 4
leak = 0.1
w_g = 8

class DiscriminatorS2(nn.Module):
    def __init__(self):
        super(DiscriminatorS2, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv3d(channels, 64, 3, stride=1, padding=(1,1,1)))

        self.conv2 = SpectralNorm(nn.Conv3d(64, 128, 4, stride=2, padding=(1,1,1)))
        self.conv3 = SpectralNorm(nn.Conv3d(128, 256, 4, stride=2, padding=(1,1,1)))
        self.conv4 = SpectralNorm(nn.Conv3d(256, 512, 4, stride=2, padding=(1,1,1)))
        #self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, padding=(1,1)))
        #self.conv4 = SpectralNorm(nn.Conv3d(128, 128, 4, stride=2, padding=(1,1,1)))
        #self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=2, padding=(1,1)))
        #self.conv6 = SpectralNorm(nn.Conv3d(256, 256, 4, stride=2, padding=(1,1,1)))
        #self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=2, padding=(1,1)))
        #self.conv8 = SpectralNorm(nn.Conv2d(512, 512*2, 3, stride=2, padding=(1,1)))
        #self.conv9 = SpectralNorm(nn.Conv2d(512*2, 512*2*2, 3, stride=2, padding=(1,1)))

        
        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512*2, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        #print(m.shape)
        #m = nn.LeakyReLU(leak)(self.conv5(m))
        #m = nn.LeakyReLU(leak)(self.conv6(m))
        #print(m.shape)
        #m = nn.LeakyReLU(leak)(self.conv7(m))
        #m = nn.LeakyReLU(leak)(self.conv8(m))
        #m = nn.LeakyReLU(leak)(self.conv9(m))
        #print(m.shape)
        out = self.fc(m.view(-1,w_g * w_g * 512*2)).view(-1)
        return out
    
class DiscriminatorS2d(nn.Module):
    def __init__(self, channels = 4):
        super(DiscriminatorS2d, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 512*2, 3, stride=2, padding=(1,1)))

        self.fc = SpectralNorm(nn.Linear(1024*4*4, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        #print(m.shape)
        m = nn.LeakyReLU(leak)(self.conv2(m))
        #print(m.shape)
        m = nn.LeakyReLU(leak)(self.conv3(m))
        #print(m.shape)
        m = nn.LeakyReLU(leak)(self.conv4(m))
        #print(m.shape)
        m = nn.LeakyReLU(leak)(self.conv5(m))
        #print(m.shape)
        #print(m.shape)
        #m = nn.LeakyReLU(leak)(self.conv5(m))
        #m = nn.LeakyReLU(leak)(self.conv6(m))
        #print(m.shape)
        #m = nn.LeakyReLU(leak)(self.conv7(m))
        #m = nn.LeakyReLU(leak)(self.conv8(m))
        #m = nn.LeakyReLU(leak)(self.conv9(m))
        #print(m.shape)
        #print('here', m.view(m.shape[0],-1).shape)
        out = self.fc(m.view(m.shape[0],-1))
        #out = self.fc(m.view(-1,w_g * w_g * 512*2)).view(-1)
        #print('out', out.shape)
        return out
    
   

class DiscriminatorS2d_sig(nn.Module):
    def __init__(self, channels = 4):
        super(DiscriminatorS2d_sig, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 512*2, 3, stride=2, padding=(1,1)))

        self.classifier = nn.Sequential(
            nn.Sigmoid()
        )
        self.fc = SpectralNorm(nn.Linear(1024*4*4, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        out = m.view(m.shape[0],-1)
        out = self.fc(out)
        out = self.classifier(out)
        #out = self.fc(m.view(-1,w_g * w_g * 512*2)).view(-1)
        #print('out', out.shape)
        return out