import sys, os, glob
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from PIL import Image
import argparse, json, torchvision
import scipy.io
import helper.canon_supervised_dataset as dset
import helper.gan_helper_fun as gh

import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn as nn
import time

    
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
def main():
    parser = argparse.ArgumentParser(description='Denoising options')
    parser.add_argument('--network', default='dvdhr', help='Choose which network to use')
    parser.add_argument('--noise_type', default='unetfourier', help='Choose which noise model to load') 
    parser.add_argument('--loss', default= 'L1_LP2D', 
                        help = 'Choose loss or a combination of losses. Options: MSE, L1, TV, LP2D (LPIPS),\
                        angular (Cosine similarity between the 4 color channels)')
    parser.add_argument('--data', default= 'stills_realvideo', help = 'Choose which data to use during training')
    parser.add_argument('--multiply', default= 'gamma', 
                        help = 'Choose what sort of processing to do on the ground truth images. Options: None, \
                        histeq (does histogram equalization on the ground truth images before taking the loss), \
                        gamma (goes gamma correction on the ground truth images, makes the network learn \
                        denoising + gamma correction')
    parser.add_argument('--space', default= 'linear') # 
    parser.add_argument('--notes', default= 'Test') 
    parser.add_argument('--crop_size', default= '512', 
                        help='Choose the image patch size for denoising. \
                        Options: 512, (512x512), full (1024x512), small (128x128), or 256 (256x256)')  
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default= 500000) 
    parser.add_argument('--preloaded', default = False, 
                        help='Use pretrained model. Specify the file path here to load in the model')
    parser.add_argument('--resume_from_checkpoint', default = False, 
                        help='Resume training from a saved checkpoint. Specify the filepath here')
    parser.add_argument('--learning_rate', default = 0.0001, type=float)
    parser.add_argument('--batch_size', default = 1, type=int)
    parser.add_argument('--save_path', default = '../saved_models/', help='Specify where to save checkpoints during training')
    parser.add_argument('--MOT_path', default = '../../work_path/MOTfiles_raw/',
                        help='If using unprocessed MOT images during training, specify the filepath \
                        for where your MOT dataset is stored')
    parser.add_argument('--stills_path', default = '../../Datasets/Canon/pairs_4_5_21_mat/', 
                        help='Specify the filepath for the stills dataset. Should be ../data/stillpaits_mat/')
    parser.add_argument('--cleanvideo_path', default = '../../work_path/Canon/8_25_21_videodataset_mat/',
                        help='Specify the filepath for the clean video dataset. Should be ../data/RGBNIR_videodataset_mat/')
    parser.add_argument('--save_every', default = 20, type=int, 
                        help='Choose save frequency. Save every N epochs. ')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    # Load in a pre-trained network before starting training
    if args.preloaded:
        print('loading in model')
        parser = argparse.ArgumentParser(description='Process some integers.')
        args1 = parser.parse_args('')
        with open(args.preloaded + '/args.txt', 'r') as f:
            args1.__dict__ = json.load(f)
        args.network = args1.network
        args.loss = args1.loss
        args.unet_opts = args1.unet_opts

    # Resume training from a saved checkpoint. 
    if args.resume_from_checkpoint:
        print('resuming from checkpoint')
        print('checkpoint filepath:', args.resume_from_checkpoint)
        folder_name = args.resume_from_checkpoint
        parser = argparse.ArgumentParser(description='Process some integers.')
        args1 = parser.parse_args('')
        with open(args.resume_from_checkpoint + '/args.txt', 'r') as f:
            args1.__dict__ = json.load(f)
            args1.resume_from_checkpoint = folder_name
        if 'n' not in args1:
            args1.nodes = args.nodes
            args1.gpus = args.gpus
            args1.nr = args.nr
            args1.world_size = args.gpus * args.nodes
            args1.batch_size = args.batch_size
        if args1.data == 'video_combined_new':
            args1.data = 'stills_realvideo_MOTvideo'
        elif args1.data == 'video_real':
            args1.data = 'stills_realvideo'
            
        args1.preloaded = False
        
        if 'crop_size' not in args1:
            args1.crop_size = 'full'
        
        args = args1
        
    # Make folder 
    base_folder = args.save_path
    folder_name = base_folder +"_".join([str(i) for i in list(args.__dict__.values())[0:7]])+'/'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(folder_name + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    args.folder_name = folder_name

    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = find_free_port() 
    
    torch.cuda.empty_cache()
    
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def get_dataset(args):
    # Load in the dataset 
    print('getting dataset')
    
    if args.crop_size == 'full':
        crop_size = (512, 1024)
        t_list =[dset.FixedCropnp(shape = crop_size), dset.ToTensor2()]
        t_list_test =[dset.FixedCropnp(shape = crop_size), dset.ToTensor2()]
        i0 = [0,0]   
    else:
        if args.crop_size == 'small':
            crop_size = (128,128)
            i0 = None
        elif args.crop_size == '256':
            crop_size = (256,256)
            i0 = None
        else:
            crop_size = (512,512)
            i0 = None
        t_list = [dset.RandCropnp(shape = crop_size), dset.ToTensor2()]
        t_list_test = [dset.FixedCropnp(shape = crop_size), dset.ToTensor2()]

    print('crop size', crop_size)
    
    # Optionally apply histogram equalization on the ground truth images (not on the noisy images)
    if 'histeq' in args.multiply:
        t_list.insert(1, dset.HistEq())
        
    # Optionally apply gamma correction on the ground truth images (not on the noisy images)
    if 'gamma' in args.multiply:
        t_list.insert(1, dset.ProcessImagePlain())
        
    composed_transforms = torchvision.transforms.Compose(t_list)
    # For simulated images, multiply the ground truth image by a value less than 1 to generate a different amplitude each time
    composed_transforms_sim = torchvision.transforms.Compose(t_list.insert(0, dset.MultiplyFixed()))
    composed_transforms_test = torchvision.transforms.Compose(t_list_test)


    dataset_list = []
    dataset_list_test = []
    
    # Load in the different types of data. Options: stills, realvideo, MOTvideo
    # Optionally load in the unprocessed MOT dataset (unpaired)
    if 'MOTvideo' in args.data:
        composed_transforms_video = torchvision.transforms.Compose([dset.MultiplyFixed(), dset.ToTensor2(), dset.RandCrop2(shape = crop_size)])
        composed_transforms_video_test = torchvision.transforms.Compose([dset.ToTensor2(), dset.FixedCrop(shape = crop_size)])

        dataset_path = args.MOT_path + 'train/*'
        base_list = glob.glob(dataset_path)
        video_dataset_dir = []
        for i in range(0,len(base_list)):
            sub_list = glob.glob(base_list[i]+'/*')
            for k in sub_list:
                video_dataset_dir.append(k)

        dataset_path = args.MOT_path + 'test/*'
        base_list = glob.glob(dataset_path)
        video_dataset_dir_test = []
        for i in range(0,len(base_list)):
            sub_list = glob.glob(base_list[i]+'/*')
            for k in sub_list:
                video_dataset_dir_test.append(k)

            
        dataset_train_video = dset.Get_sample_batch_simvideo_distributed2(video_dataset_dir, 
                                                                          composed_transforms_video, 
                                                                          crop_size = crop_size)
        dataset_test_video = dset.Get_sample_batch_simvideo_distributed2(video_dataset_dir_test, 
                                                                         composed_transforms_video_test, 
                                                                         start_ind=0, crop_size = crop_size)
        # add to dataset list
        dataset_list.append(dataset_train_video)
        dataset_list_test.append(dataset_test_video)
        
    # Optionally load in the real stills dataset (paired)
    if 'stills' in args.data:
        all_files_mat = glob.glob(args.stills_path + '*.mat')[0:59]
        all_files_mat_test = glob.glob(args.stills_path + '*.mat')[59:]

        dataset_train = dset.Get_sample_batch(all_files_mat, composed_transforms)
        dataset_test = dset.Get_sample_batch(all_files_mat_test, composed_transforms_test, start_ind=0)
    
        # add to dataset list
        dataset_list.append(dataset_train)
        dataset_list_test.append(dataset_test)
    
    # Optionally load in the real clean video dataset (unpaired)
    if 'realvideo' in args.data:
        
        all_real_videos = np.sort(glob.glob(args.cleanvideo_path + '*'))
        file_path_list = []
        file_path_list_test = []
        for k in range(0,len(all_real_videos)):
            all_files = glob.glob(all_real_videos[k] +'/*.mat')
            inds = []
            for i in range(0, len(all_files)):
                inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))

            inds_sort = np.argsort(inds)
            all_files_sorted = np.array(all_files)[inds_sort]
            sublist = all_files_sorted[0::16][0:-2]
            file_path_list_test.append(all_files_sorted[0::16][-2:-1][0])
            for i in sublist:
                file_path_list.append(i)
        
        dataset_real_videos = dset.Get_sample_batch_video_distributed2(file_path_list, composed_transforms)
        dataset_real_videos_test = dset.Get_sample_batch_video_distributed2(file_path_list_test, composed_transforms_test)

        # add to dataset list
        dataset_list.append(dataset_real_videos)
        dataset_list_test.append(dataset_real_videos_test)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test, i0

def get_model(args):
# Define the denoising model 
    if args.network == 'Unet3D':
        from models.Unet3d import Unet3d
        res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1]) 
        model = Unet3d(n_channel_in=4, 
                     n_channel_out=4, 
                     residual=res_opt, 
                    down=args.unet_opts.split('_')[1], 
                     up=args.unet_opts.split('_')[2], 
                     activation=args.unet_opts.split('_')[3])
    elif args.network == 'DnCNN3D':
        from models.dncnn3d import DnCNN
        model = DnCNN(channels = 4)
    elif args.network == 'hrnet2d':
        import helper.hr_helper as hr
        model = hr.load_2d_hrnet(num_channels=64)
    elif args.network == 'hrnet3d':
        import helper.hr_helper as hr
        model = hr.load_3d_hrnet(num_channels=4)
    elif args.network =='dvdnet':
        from models.fastdvdnet import FastDVDnet
        model = FastDVDnet()
        print('loading FastDVDnet')
    elif args.network =='dvdhr':
        from models.fastdvdnet import FastDVDnetHR
        model = FastDVDnetHR()
        print('loading FastDVDnet HR')
    elif args.network =='dvdhrie':
        from models.fastdvdnet import FastDVDnetHRie
        model = FastDVDnetHRie()
        print('loading FastDVDnet HRie')    
    elif args.network =='dvdhr16':
        from models.fastdvdnet import FastDVDnetHR16
        model = FastDVDnetHR16()
        print('loading FastDVDnet HR 16')
    else:
        print('Error, invalid network')
        
    return model

# Load in a pretrained model from a checkpoint
def preload_model(args, model, device):
    print(args.preloaded)
    list_of_files = glob.glob(args.preloaded + '/checkpoint*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
   
    saved_state_dict = torch.load(path, map_location = 'cuda:'+str(device))
    
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

    curr_epoch = int(path.split('/')[-1].split('_')[0].split('checkpoint')[1])
    print('resuming from preloaded, epoch:', curr_epoch)
    return model

# Resume training from a checkpoint
def resume_from_checkpoint(args, model, device):
    list_of_files = glob.glob(args.resume_from_checkpoint + '/checkpoint*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location = 'cuda:'+str(device))
    
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
    
    curr_epoch = int(path.split('/')[-1].split('_')[0].split('checkpoint')[1])
    loaded = scipy.io.loadmat(args.resume_from_checkpoint + '/losses.mat')
    loss_list = list(loaded['train_loss'][0])
    test_loss_list = list(loaded['test_loss'][0])
    folder_name = args.resume_from_checkpoint + '/'
    print('resuming from checkoint, epoch:', curr_epoch)
    return model, curr_epoch, loss_list, test_loss_list, folder_name

def load_generator_model(args, gpu):
    if args.noise_type == 'unetfourier': 
        generator = gh.load_generator2('../saved_models/noise_generator', gpu)
    # Can add conditionals for alternative noise generators here
    elif args.noise_type == 'shotreaduniform': 
        base_file = '../../work_path/starlight_denoising_saved_models_smaller/generator/'
        generator = gh.load_from_checkpoint_ab(base_file +'noUnet_newcalib_color_gray_True_periodic_lpips_mixed_patches_after_shot_read_uniform_256_Oct29_ablation', device = gpu)
    elif args.noise_type == 'ulpf': 
        base_file = '../../work_path/starlight_denoising_saved_models_smaller/generator/'
        generator = gh.load_from_checkpoint_ab(base_file +'Unet_newcalib_color_gray_True_periodic_lpips_fourier_patches_after_shot_read_uniform_row1_rowt_learnedfixed_periodic_256_Nov7fixedlearned2', device = gpu)
    else:
        print('invalid generator')
    return generator#.cuda(gpu)
    
def define_loss(args, gpu):
    all_losses = []
    if 'MSE' in args.loss:
        print('using MSE loss')
        all_losses.append(MSELoss().cuda(gpu))
    if 'L1' in args.loss:
        print('using L1 loss')
        all_losses.append(L1Loss().cuda(gpu))
    if 'TV' in args.loss:
        print('using TV loss')
        loss_tv = lambda a,b: 1e-6*gh.tv_loss(a)
        all_losses.append(loss_tv.cuda(gpu))
    if 'LP2D' in args.loss:
        print('using LPIPS loss')
        import lpips
        loss_lpips1 = lpips.LPIPS(net='alex').cuda(gpu)
        if 'ccm' in args.space:
            loss_lpips = lambda a,b: torch.sum(1e-1*loss_lpips1(ccm(a),ccm(b)))
        else:
            loss_lpips = lambda a,b: torch.sum(1e-1*loss_lpips1(a[:,0:3],b[:,0:3]))
        all_losses.append(loss_lpips)
    if 'LPIPS' in args.loss:
        print('using LPIPS loss')
        import lpips
        loss_lpips1 = lpips.LPIPS(net='alex').cuda(gpu)

        def loss_lpips(a,b):
            sz = a.shape

            a_new = a.reshape(sz[0]*sz[2], sz[1], sz[3], sz[4])
            b_new = b.reshape(sz[0]*sz[2], sz[1], sz[3], sz[4])

            final_loss = torch.sum(1e-1*loss_lpips1(a_new[:,0:3],
                     b_new[:,0:3]))
            return final_loss

        all_losses.append(loss_lpips)
        
    if 'angular' in args.loss:
        cos_between = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        angular_loss = lambda a,b: torch.mean(1e-1*torch.acos(torch.clamp(cos_between(a,b),-0.99999, 0.99999))*180/np.pi) 
        
        all_losses.append(angular_loss)

    loss_function = lambda a,b: torch.sum(torch.stack([torch.sum(all_losses[i](a,b)) for i in range(0, len(all_losses))]))
    
    return loss_function

def run_test(gpu, args, test_loader, model, criterion, generator):
    print('running test')

    with torch.no_grad():
        avg_test_loss = 0
        for i, sample in enumerate(test_loader): #Loop through dataset
            gt_label = sample['gt_label_nobias'].cuda(non_blocking=True)
            
            if sample['noisy_input'][0,0,0,0,0]<-3:
                with torch.no_grad():
                    net_input = gh.t23_1(generator(gh.t32_1(sample['gt_label_nobias'].cuda(non_blocking=True))))
            else:
                net_input = sample['noisy_input'].cuda(non_blocking=True)

            
            if args.network == 'hrnet2d':
                szo = net_input.shape
                net_output = model(net_input.reshape(szo[0],szo[1]*szo[2],szo[3],szo[4])).reshape(szo)
            elif args.network =='dvdnet' or args.network =='dvdhr' or args.network =='dvdhrie':
                curr_ind = 8
                net_output = model(net_input[:,:,curr_ind-2:curr_ind+3])
                gt_label = gt_label[:,:,curr_ind]
            elif args.network == 'dvdhr16':
                net_output = model(net_input)
                gt_label = gt_label[:,:,8]
            else:
                net_output = model(net_input)

                
            test_loss = criterion(net_output, gt_label)
            avg_test_loss += test_loss.item()

            
        avg_test_loss = avg_test_loss/(i+1)    
        
        if args.network =='dvdnet' or args.network =='dvdhr' or args.network == 'dvdhr16' or args.network =='dvdhrie':
            out_plt = net_output.cpu().detach().numpy()[0,].transpose(1,2,0)[...,0:3]
        else:
            out_plt = net_output.cpu().detach().numpy()[0,:,8].transpose(1,2,0)[...,0:3]

        
    return avg_test_loss, out_plt

def save_checkpoint(folder_name, ep, test_loss_list, train_loss, best_test_loss, model, out_plt):
    
    print('saving checkpoint')
    checkpoint_name = folder_name + f'checkpoint{ep}_test_loss{test_loss_list[-1]:.5f}_trainloss{np.round(train_loss[-1], 5)}.pt'
    torch.save(model.state_dict(), checkpoint_name)
    save_name = folder_name + f'testimage{ep}_test_loss{test_loss_list[-1]:.5f}_trainloss{np.round(train_loss[-1], 2)}.jpg'
    Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)
    scipy.io.savemat(folder_name + 'losses.mat', {'train_loss': train_loss, 'test_loss': test_loss_list})
    
    if best_test_loss > test_loss_list[-1]:
            print('best loss', test_loss_list[-1])
            best_test_loss = test_loss_list[-1]
            checkpoint_name = folder_name + f'best{ep}_test_loss{test_loss_list[-1]:.5f}.pt'
            torch.save(model.state_dict(), checkpoint_name)
            save_name = folder_name + f'best{ep}_test_loss{test_loss_list[-1]:.5f}.jpg'
            Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

    
# Main training function 
def train(gpu, args):
    print('entering training function')
    print(args.nr, args.gpus, gpu, args.world_size)
    # Setup for distributed training on multiple GPUs
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank)    
    
    
    print('loading model')
    model = get_model(args)
    
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    
    if args.preloaded:
        model = preload_model(args, model, gpu)
    
    if args.resume_from_checkpoint:
        print('resuming model from checkpoint')
        model, curr_epoch, train_loss, test_loss_list, folder_name = resume_from_checkpoint(args, model, gpu)
            
        best_test_loss = test_loss_list[-1]
        print('best loss is: ', best_test_loss)
    else:
        curr_epoch = 0; train_loss = []; test_loss_list = []; best_test_loss = 1e9
        folder_name = args.folder_name
        
    
    print('loading generator', gpu)
    generator = load_generator_model(args, gpu)
    generator.cuda(gpu)
    
    batch_size = args.batch_size
    
    criterion = define_loss(args, gpu)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    
    # Set up dataset
    dataset_list, dataset_list_test, i0 = get_dataset(args)
    print('i0', i0)
    generator.indices = i0 
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_list, 
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_list, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler) 
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_list_test, 
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=test_sampler) 
    
    

    total_step = len(train_loader)

    print('Enter training loop')
    for ep in range(curr_epoch, args.num_iter):
        
        avg_loss = 0
        for i, sample in enumerate(train_loader):
            start = time.time()
            
            gt_label = sample['gt_label_nobias'].cuda(non_blocking=True)
            
            # If using simulated data, generate the noisy video clip using the noise generator
            if sample['noisy_input'][0,0,0,0,0]<-3:
                with torch.no_grad():
                    net_input = gh.t23_1(generator(gh.t32_1(sample['gt_label_nobias'].cuda(non_blocking=True))))
            # Otherwise, use the real noisy clip
            else:
                net_input = sample['noisy_input'].cuda(non_blocking=True)
          
            
            if args.network == 'hrnet2d':
                szo = net_input.shape
                net_output = model(net_input.reshape(szo[0],szo[1]*szo[2],szo[3],szo[4])).reshape(szo)
            elif args.network =='dvdnet' or args.network =='dvdhr' or args.network =='dvdhrie':
                curr_ind = 8
                net_output = model(net_input[:,:,curr_ind-2:curr_ind+3])
                gt_label = gt_label[:,:,curr_ind]
            elif args.network == 'dvdhr16':
                net_output = model(net_input)
                gt_label = gt_label[:,:,8]
            else:
                net_output = model(net_input)

                
            loss = criterion(net_output, gt_label)
            
            # Backward and optimize
            start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss+=loss.item()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        ep + 1, 
                        args.num_iter, 
                        i + 1, 
                        total_step,
                        loss.item())
                       )

       
        train_loss.append(avg_loss/i)     
        if ep%args.save_every == 0 and gpu == 0:
            avg_test_loss, out_plt = run_test(gpu, args, test_loader, model, criterion, generator)
            test_loss_list.append(avg_test_loss)
            save_checkpoint(folder_name, ep, test_loss_list, train_loss, best_test_loss, model, out_plt)
        
                
if __name__ == '__main__':
    main()