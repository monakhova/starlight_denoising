import sys, os, glob
sys.path.append("../.")
sys.path.append("../data/")
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
import lpips


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
    parser = argparse.ArgumentParser(description='Gan noise model training options.')
    parser.add_argument('--network', default='Unet', help = 'Options: Unet, Unet_cat, noUnet, Unet_first') 
    parser.add_argument('--noiselist', default='shot_read_uniform_row1_rowt_fixed1_periodic', 
                        help = 'Specify the type of noise to include. \
                        Options: read, shot, uniform, row1, rowt, fixed1, learnedfixed, periodic') 
    parser.add_argument('--crop_size', default=256, type = int) 
    parser.add_argument('--dataset', default='color_gray', help = 'Choose which dataset to use. Options: gray, color')
    parser.add_argument('--discriminator_loss', default='fourier', 
                        help = 'Choose generator loss. Options: mixed, fourier, real, mean') 
    parser.add_argument('--notes', default= 'yournamehere') 
    
    parser.add_argument('--generator_loss', default='lpips', help = 'Choose generator loss. Default: lpips') 
    parser.add_argument('--split_into_patches', default='patches_after') 
    parser.add_argument('--save_path', default = '../saved_models/', help='Specify where to save checkpoints during training') 
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default= 500000) 
    parser.add_argument('--device', default= 'cuda:0')
    parser.add_argument('--lr', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', default = 1, type=int)

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

    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = find_free_port() 
    
    torch.cuda.empty_cache()
    
    folder_name = args.save_path + 'noisemodel' +"_".join([str(i) for i in list(args.__dict__.values())[0:6]])+'/'

   
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(folder_name + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    args.folder_name = folder_name
        
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    

def get_dataset(args):    
    composed_transforms = torchvision.transforms.Compose([dset.ToTensor2(), dset.AddFixedNoise(), dset.RandCrop_gen(shape = (args.crop_size,args.crop_size))])
    composed_transforms2 = torchvision.transforms.Compose([dset.ToTensor2(), dset.FixedCrop_gen(shape = (args.crop_size,args.crop_size))])


    dataset_list = []
    dataset_list_test = []
    
    if 'gray' in args.dataset:
        filepath_noisy = '../data/paired_data/graybackground_mat/'
        dataset_train_gray = dset.Get_sample_noise_batch(filepath_noisy, composed_transforms, fixed_noise = False)
        
        dataset_list.append(dataset_train_gray)
        
        if 'color' not in args.dataset:
            all_files_mat_test = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[40:]
            dataset_test_real = dset.Get_sample_batch(all_files_mat_test, composed_transforms)
            dataset_list_test.append(dataset_test_real)
        
    if 'newcalib' in args.dataset:
        
        filepath_noisy = '../data/paired_data/colorbackground_mat/'
        
        filepath_noisy1 = glob.glob(filepath_noisy + '*')[1:-2]
        filepath_noisy2 = [glob.glob(filepath_noisy + '*')[0], glob.glob(filepath_noisy + '*')[-1]]
        
        dataset_train_gray2 = dset.Get_sample_noise_batch_new(filepath_noisy1, composed_transforms)
        dataset_test_gray2 = dset.Get_sample_noise_batch_new(filepath_noisy2, composed_transforms)
                            
        dataset_list.append(dataset_train_gray2)
        dataset_list_test.append(dataset_test_gray2)
        
        
    if 'color' in args.dataset:
        all_files_mat = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[0:40]
        all_files_mat_test = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[40:]
    
        dataset_train_real = dset.Get_sample_batch(all_files_mat, composed_transforms)
        dataset_test_real = dset.Get_sample_batch(all_files_mat_test, composed_transforms2)
        
        dataset_list.append(dataset_train_real)
        dataset_list_test.append(dataset_test_real)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test
    
def get_model(args, device):
    from models.unet import Unet

    if args.network == 'noUnet':
        model = None 
    else:
        if args.network == 'Unet_cat':
            in_channels = 8
        else:
            in_channels = 4

        if 'newunet' in args.network:
            import models.fastdvdnet as fdvd
            model = fdvd.DenBlockUnet(num_input_frames=1).to(args.device)
            model.weight_init
            for param in model.parameters():
                param.data = param.data*1e-6
            
        else:
            res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1]) 
            model = Unet(n_channel_in=in_channels, 
                         n_channel_out=4, 
                         residual=res_opt, 
                        down=args.unet_opts.split('_')[1], 
                         up=args.unet_opts.split('_')[2], 
                         activation=args.unet_opts.split('_')[3])
        
        
            
    if args.discriminator_loss == 'mean' or args.discriminator_loss == 'complex' or args.discriminator_loss == 'mixed':
        disc_channels = 8
    else:
        disc_channels = 4

    # old version: 
    # discriminator = gh.DiscriminatorS2d().to(args.device)
    discriminator = gh.DiscriminatorS2d_sig(channels = disc_channels)
    
    # old version: 
    #generator = gh.NoiseGenerator2d3d(net = model, unet_opts = args.network, add_fixed = args.addfixed)
    generator = gh.NoiseGenerator2d3d_distributed_ablation(net = model, unet_opts = args.network, noise_list = args.noiselist, 
                                               device = device)
    
    return generator, discriminator


def define_loss(args, gpu):
    print('using lpips loss')
    loss_fn_alex = lpips.LPIPS(net='alex').to(gpu)
    def gen_loss(in1, in2): 
        
        total_loss = 0
        if in1.shape[1]==8:
            total_loss+=torch.mean(loss_fn_alex(in1[:,0:3],
                                                in2[:,0:3],0,1))
            total_loss+=torch.mean(loss_fn_alex(in1[:,4:7],
                                                in2[:,4:7],0,1))
        else:
            total_loss+=torch.mean(loss_fn_alex(in1[:,0:3],
                                                in2[:,0:3],0,1))
        
        return total_loss
        
    return gen_loss, loss_fn_alex
    

    
def train(gpu, args):
    print('entering training function')
    
    print(args.nr, args.gpus, gpu, args.world_size)
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank)    
    
    
    print('loading model')
    generator, discriminator = get_model(args, gpu)
    
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    generator.cuda(gpu)
    discriminator.cuda(gpu)
    
    folder_name = args.folder_name
    
    batch_size = args.batch_size
    
    gen_loss, loss_fn_alex = define_loss(args, gpu)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Wrap the model
    generator = nn.parallel.DistributedDataParallel(generator,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    # Wrap the model
    discriminator = nn.parallel.DistributedDataParallel(discriminator,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    # Set up dataset
    dataset_list, dataset_list_test = get_dataset(args)
    
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
    
    
    
    if args.split_into_patches == 'patches_before':
        split_patches = True
    else:
        split_patches = False
    


    ## WGAN-GP
    n_critic = 5
    lambda_gp = 10
    num_epochs = 5000
    G_losses = []
    D_losses = []
    kld_list = []
    real_list = []
    fake_list = []


    best_kld = 1e6
    for epoch in range(0,num_epochs):
        for i, sample in enumerate(train_loader):

            noisy_raw = torch.transpose(sample['noisy_input'],0, 2).squeeze(2).to(gpu)
            clean_raw = torch.transpose(sample['gt_label_nobias'],0, 2).squeeze(2).to(gpu)

            generator.indices = sample['rand_inds']
            # -----------------
            #  Train Discriminator 
            # -----------------
            ## Train with batch

            optimizer_D.zero_grad()

            # Generator fake noisy images 
            gen_noisy = generator(clean_raw, split_patches)
            if args.discriminator_loss == 'mean':
                gen_mean = torch.mean(gen_noisy,0).unsqueeze(0)
                real_mean = torch.mean(noisy_raw,0).unsqueeze(0)

                gen_noisy = torch.cat((gen_mean.repeat(16,1,1,1), gen_noisy),1)
                noisy_raw = torch.cat((real_mean.repeat(16,1,1,1), noisy_raw),1)


            if split_patches == False:
                gen_noisy = gh.split_into_patches2d(gen_noisy).to(gpu)


            real_noisy = gh.split_into_patches2d(noisy_raw).to(gpu)
            clean = gh.split_into_patches2d(clean_raw).to(gpu)

            if 'fourier' in args.discriminator_loss:
                #print('using fourier loss for discriminator')
                real_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho")))
                gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

            elif 'mixed' in args.discriminator_loss:
                #print('using fourier + real loss for discriminator')
                real_noisy1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho")))
                gen_noisy1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

                real_noisy = torch.cat((real_noisy, real_noisy1),1)
                gen_noisy = torch.cat((gen_noisy, gen_noisy1),1)

            elif 'complex' in args.discriminator_loss:
                #print('using fourier complex loss for discriminator')
                real_noisy1 = torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho"))
                gen_noisy1 = torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho"))

                real_noisy = torch.cat((torch.real(real_noisy1), torch.imag(real_noisy1)),1)
                gen_noisy = torch.cat((torch.real(gen_noisy1), torch.imag(gen_noisy1)),1)

                
            real_validity = discriminator(real_noisy)
            fake_validity = discriminator(gen_noisy)


            # Gradient penalty
            gradient_penalty = gh.compute_gradient_penalty2d(discriminator, real_noisy.data, gen_noisy.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()


            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(clean_raw, split_patches)
                if args.discriminator_loss == 'mean':
                    fake_imgs_mean = torch.mean(fake_imgs,0).unsqueeze(0)
                    fake_imgs = torch.cat((fake_imgs_mean.repeat(16,1,1,1), fake_imgs),1)
                
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                if split_patches == False:
                    fake_imgs = gh.split_into_patches2d(fake_imgs).to(gpu)
                
                if 'fourier' in args.discriminator_loss:
                    #print('using fourier loss for discriminator')
                    fake_imgs = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho")))
                elif 'mixed' in args.discriminator_loss:
                    #print('using mixed loss for discriminator')
                    fake_imgs1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho")))
                    fake_imgs = torch.cat((fake_imgs, torch.abs(fake_imgs1)), 1)
                elif 'complex' in args.discriminator_loss:
                    #print('using mixed loss for discriminator')
                    fake_imgs1 = torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho"))

                    fake_imgs = torch.cat((torch.real(fake_imgs1), torch.imag(fake_imgs1)),1)


                fake_validity = discriminator(fake_imgs)
                
                g_loss = -torch.mean(fake_validity)


                if args.generator_loss == 'lpips':
                    g_loss += gen_loss(fake_imgs, real_noisy) 


                g_loss.backward()
                optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f]"
            % (epoch, num_epochs, i, d_loss.item(), g_loss.item())
        )

        gen1 = (gen_noisy).detach().cpu().numpy()
        real1 = (real_noisy).detach().cpu().numpy()
        kld_val = gh.cal_kld(gen1, real1)
        print('KLD', kld_val)

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        kld_list.append(kld_val)

        real_list.append(torch.mean(real_validity).item())
        fake_list.append(torch.mean(fake_validity).item())


        # Check if Best KLD value
        if epoch % 5 == 0:
            tot_kld = 0
            for i, sample in enumerate(test_loader):

                with torch.no_grad():
                    noisy_raw = torch.transpose(sample['noisy_input'],0, 2).squeeze(2)
                    clean_raw = torch.transpose(sample['gt_label_nobias'],0, 2).squeeze(2).to(gpu)

                    generator.indices = [10,10]

                    gen_noisy = generator(clean_raw, split_patches)

                    if split_patches == False:
                        gen_noisy = gh.split_into_patches2d(gen_noisy).to(gpu)

                    real_noisy = gh.split_into_patches2d(noisy_raw).to(gpu)
                    clean = gh.split_into_patches2d(clean_raw).to(gpu)

                    gen1 = (gen_noisy).detach().cpu().numpy()
                    real1 = (real_noisy).detach().cpu().numpy()
                    kld_val = gh.cal_kld(gen1, real1)
                    tot_kld += kld_val

            print('Total KLD value:', tot_kld)

            if tot_kld < best_kld:
                best_kld = tot_kld

                print('saving best')
                checkpoint_name = folder_name + f'bestgenerator{epoch}_KLD{best_kld:.5f}.pt'
                torch.save(generator.state_dict(), checkpoint_name)

                checkpoint_name = folder_name + f'bestdiscriminatort{epoch}_KLD{best_kld:.5f}.pt'
                torch.save(discriminator.state_dict(), checkpoint_name)


            if gpu==0:
                print('saving checkpoint')

                out_plt = gen_noisy.cpu().detach().numpy()[0].transpose(1,2,0)[...,0:3]

                checkpoint_name = folder_name + f'generatorcheckpoint{epoch}_Gloss{G_losses[-1]:.5f}_Dloss{np.round(D_losses[-1], 5)}.pt'
                torch.save(generator.state_dict(), checkpoint_name)

                checkpoint_name = folder_name + f'discriminatorcheckpoint{epoch}_Gloss{G_losses[-1]:.5f}_Dloss{np.round(D_losses[-1], 5)}.pt'
                torch.save(discriminator.state_dict(), checkpoint_name)

                save_name = folder_name + f'testimage{epoch}.jpg'
                Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

                scipy.io.savemat(folder_name + 'losses.mat',
                                {'G_losses':G_losses, 
                                 'D_losses':D_losses,
                                'kld_list':kld_list,
                                'real_list':real_list,
                                'fake_list':fake_list})
        
if __name__ == '__main__':
    main()