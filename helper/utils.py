import torch
import argparse, json, glob, os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from ipywidgets import interact, widgets, fixed

def plotf2(r, img1, ttl, sz):

    plt.title(ttl+' {}'.format(r))
    plt.imshow(img1[r][...,0:3], cmap="gray", vmin = 0, vmax = np.max(img1));
    fig = plt.gcf()
    fig.set_size_inches(sz)
    plt.show();

    return 

def plt3D(img1, title = '', size = (15,15)):
    interact(plotf2, 
             r=widgets.IntSlider(min=0,max=np.shape(img1)[0]-1,step=1,value=1), 
             img1 = fixed(img1),
             continuous_update= False, 
             ttl = fixed(title), 
             sz = fixed(size));

    
    
def load_video_seq(folder_name, seq_id, start_ind, num_to_load):
    base_name_seq = folder_name + 'seq' + str(seq_id) + '/'

    filepaths_all = glob.glob(base_name_seq + '*.mat')
    total_num = len(filepaths_all)

    ind = []
    for i in range(0,len(filepaths_all)):
        ind.append(int(filepaths_all[i].split('/')[-1].split('.')[0]))
    ind = np.argsort(np.array(ind))
    filepaths_all_sorted = np.array(filepaths_all)[ind]

    if num_to_load == 'all':
        num_to_load = total_num
        print('loading ', num_to_load, 'frames')
    full_im = np.empty((num_to_load, 640, 1080, 4))
    for i in range(0,num_to_load):
        loaded = scipy.io.loadmat(filepaths_all_sorted[start_ind +i])
        full_im[i] = loaded['noisy_list'].astype('float32')/2**16

    return full_im

def run_denoiser(sample, args_list, models_to_test, device):
    i=0
    with torch.no_grad():
            net_input = sample['noisy_input'].to(device)
            if args_list[i].network == 'hrnet2d':
                    szo = net_input.shape
                    net_output = models_to_test[i](net_input.to(device).reshape(szo[0],szo[1]*szo[2],szo[3],szo[4])).reshape(szo)
            elif args_list[i].network =='dvdnet' or args_list[i].network =='dvdhr':
                net_output = torch.zeros_like(net_input[:,:,2:-3])
                for j in range(2,sample['noisy_input'].shape[2]-3):
                    net_input = sample['noisy_input'].to(device)
                    curr_ind = j
                    net_output[:,:,j-2] = models_to_test[i](net_input[:,:,curr_ind-2:curr_ind+3].to(device))
            else:
                net_output = models_to_test[i](net_input.to(device))

            out_plt = net_output.cpu().detach().numpy()[0].transpose(1,2,3,0)

    return out_plt

def load_from_checkpoint(folder_name, best = True):
    device = 'cuda:0'
    print('loading from checkpoint')
    parser = argparse.ArgumentParser(description='Process some integers.')
    args1 = parser.parse_args('')
    with open(folder_name + '/args.txt', 'r') as f:
        args1.__dict__ = json.load(f)
        args1.fraction_video = 50
        args1.resume_from_checkpoint = folder_name
    args = args1
    
    if args.network == 'Unet3D':
        from models.Unet3d import Unet3d
        res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1]) 
        model = Unet3d(n_channel_in=4, 
                     n_channel_out=4, 
                     residual=res_opt, 
                    down=args.unet_opts.split('_')[1], 
                     up=args.unet_opts.split('_')[2], 
                     activation=args.unet_opts.split('_')[3]).to(device)
    elif args.network == 'DnCNN3D':
        from models.dncnn3d import DnCNN
        model = DnCNN(channels = 4).to(device)
    elif args.network == 'hrnet2d':
        import hr_helper as hr
        model = hr.load_2d_hrnet(num_channels=64).to(device)
    elif args.network == 'hrnet3d':
        import hr_helper as hr
        model = hr.load_3d_hrnet(num_channels=4).to(device)
    elif args.network =='dvdnet':
        from models.fastdvdnet import FastDVDnet
        model = FastDVDnet()
        model.to(device)
        print('loading FastDVDnet')
    elif args.network =='dvdhr':
        from models.fastdvdnet import FastDVDnetHR
        model = FastDVDnetHR()
        model.to(device)
        print('loading FastDVDnet HR')
    elif args.network =='dvdhr16':
        from models.fastdvdnet import FastDVDnetHR16
        model = FastDVDnetHR16()
        model.to(device)
        print('loading FastDVDnet HR')
    elif args.network =='litehrnet3d':
        import models.litehrnet3d as ll
        dict_opts = dict(stem = dict(stem_channels = 4,
                           out_channels = 4,
                           expand_ratio = 1),
                num_stages = 3,

                stages_spec=dict(
                    num_modules=(2, 4, 2),
                    num_branches=(2, 3, 4),
                    num_blocks=(2, 2, 2),
                    module_type=('LITE', 'LITE', 'LITE'),
                    with_fuse=(True, True, True),
                    reduce_ratios=(1, 1, 1),
                    num_channels=(
                        (4, 80),
                        (4, 80, 160),
                        (4, 80, 160, 320),
                    )),
                     with_head=False,
                    )

        model = ll.LiteHRNet1(dict_opts, in_channels=4, zero_init_residual=True).to(device)
        model.init_weights()
    else:
        print('Error, invalid network')
        
    if best == True:
        list_of_files = glob.glob(folder_name + '/best*.pt') # * means all if need specific format then *.csv
    else:
        list_of_files = glob.glob(folder_name + '/checkpoint*.pt') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location = device)
    
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
        
    model.load_state_dict(saved_state_dict)
    
    if best == True:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('best')[1])
    else:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('checkpoint')[1])
    loaded = scipy.io.loadmat(args.resume_from_checkpoint + '/losses.mat')
    loss_list = list(loaded['train_loss'][0])
    test_loss_list = list(loaded['test_loss'][0])
    folder_name = args.resume_from_checkpoint + '/'
    print('resuming from checkoint, epoch:', curr_epoch)
    
    return args, model