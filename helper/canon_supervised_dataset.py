from helper.canon_utils import read_16bit_raw, raw_to_4
import torch
import sys, os, glob
import numpy as np
import scipy.io
#sys.path.append("../Datasets/Canon")
from PIL import Image
import helper.post_processing as pp
import time
import cv2
from skimage import exposure
import torchvision
from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


def get_dataset_noise_visualization(dataset_arg):  
    crop_size = 512
    composed_transforms = torchvision.transforms.Compose([ToTensor2(), RandCrop_gen(shape = (crop_size,crop_size))])
    composed_transforms2 = torchvision.transforms.Compose([ToTensor2(), FixedCrop_gen(shape = (crop_size,crop_size*2))])


    dataset_list = []
    dataset_list_test = []
    
    if 'gan_gray' in dataset_arg:
        filepath_noisy = 'data/paired_data/graybackground_mat/'
        dataset_train_gray = Get_sample_noise_batch(filepath_noisy, composed_transforms, fixed_noise = False)
        dataset_list_test.append(dataset_train_gray)
        
    if 'gan_color' in dataset_arg:
        filepath_noisy = 'data/paired_data/colorbackground_mat/'
        filepath_noisy = glob.glob(filepath_noisy + '*')[1:-2]
        
        dataset_test_gray2 = Get_sample_noise_batch_new(filepath_noisy, composed_transforms)                
        dataset_list_test.append(dataset_test_gray2)
        
    if 'natural' in dataset_arg:
        all_files_mat_test = glob.glob('data/paired_data/stillpairs_mat/*.mat')[40:-1]
        dataset_test_real = Get_sample_batch(all_files_mat_test, composed_transforms2)
        dataset_list_test.append(dataset_test_real)
        
    if len(dataset_list_test)>1:
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list_test

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        print('converting to tensor')
        for key in sample:
            if not isinstance(sample[key], int):
                if len(sample[key].shape) == 3:
                    sample[key] = torch.tensor(sample[key].transpose(2,0,1).copy(), dtype = torch.float32).unsqueeze(0)
                elif len(sample[key].shape) == 4:
                    sample[key] = torch.tensor(sample[key].transpose(3,0,1,2).copy(), dtype = torch.float32).unsqueeze(0)
        print('done converting to tensor')
        return sample
    
class ProcessImage(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, processing_steps = [pp.ccm_3x4, pp.clip, pp.gamma]):
        self.processing = processing_steps
    def __call__(self, sample):
    
        for key in sample:
            if 'gt' in key:
                sample[key][...,0:3] = pp.process(sample[key], self.processing)
        return sample
    
class ProcessImagePlain(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, processing_steps = [pp.clip, pp.gamma]):
        self.processing = processing_steps
    def __call__(self, sample):
    
        for key in sample:
            if 'gt' in key:
                sample[key] = pp.process(sample[key], self.processing)
        return sample

class HistEq(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
    
        for key in sample:
            if 'gt' in key:
                sample[key] = exposure.equalize_hist(sample[key])
        return sample
    
class MultiplyFixed(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, low=0.1, high=1):
        self.low = low
        self.high = high
    def __call__(self, sample):
    
        const = np.random.uniform(self.low, self.high)
        for key in sample:
            sample[key] = sample[key]*const
        return sample
    
class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        for key in sample:
            #if not sample[key].shape[-1] ==1:
            if len(sample[key].shape) == 3:
                sample[key] = torch.tensor(sample[key].transpose(2,0,1), dtype = torch.float32)
            elif len(sample[key].shape) == 4:
                sample[key] = torch.tensor(sample[key].transpose(3,0,1,2), dtype = torch.float32)
        return sample
    
class AddFixedNoise(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
            mean_noise = scipy.io.loadmat('../data/fixed_pattern_noise.mat')['mean_pattern']
            self.fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(self.fixed_noise.transpose(2,0,1).copy(), dtype = torch.float32).unsqueeze(1)
    def __call__(self, sample):
        for key in sample:
                if key == 'gt_label':
                    sample[key] = sample[key] + self.fixednoiset[0]
        return sample
    
class AddFixedNoise2(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
            mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
            self.fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(self.fixed_noise.transpose(2,0,1).copy(), dtype = torch.float32).unsqueeze(1)
    def __call__(self, sample):
        for key in sample:
            
                if key == 'gt_label':

                    i1 = np.random.randint(0, self.fixednoiset.shape[-2] - sample[key].shape[-2])
                    i2 = np.random.randint(0, self.fixednoiset.shape[-1] - sample[key].shape[-1])
                    sample[key] += self.fixednoiset[...,i1:i1+sample[key].shape[-2], i2:i2 + sample[key].shape[-1]]
                
        return sample
    
class RandFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
    
        if np.random.randint(2, size=1)[0] == 1:  # random flip  
            for key in sample:
                if not sample[key].shape[-1] ==1:
                    sample[key] = torch.flip(sample[key], dims=[-1])
                
        if np.random.randint(2, size=1)[0] == 1:  # random flip  
            for key in sample:
                if not sample[key].shape[-1] ==1:
                    sample[key] = torch.flip(sample[key], dims=[-2])

        return sample    
    
class CCM(object):
    def __init__(self, device):
        
        myccm = np.array([[ 0.76031811,  0.19460622, -0.09200754, -0.04863701],
           [-0.30808756,  1.67370372, -0.08039811, -0.73159016],
           [ 0.2734654 , -0.53701519,  2.24788416, -1.26116684]])
        
        self.ccmt = torch.tensor(myccm, dtype = torch.float32, device = device)
        
    def __call__(self, im):
        
        orig_shape = im.shape
        out_im = torch.mm(self.ccmt, im.squeeze(0).reshape(4,-1))
        

        return out_im.reshape((orig_shape[0], 3, *orig_shape[2:]))
    
class UnetCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        for key in sample:
            if sample[key] is not None:
                sample[key] = sample[key][...,0:512,0:512*2]
        
        return sample
    
class FixedCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = 0
        i1 = 0
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1]]
        
        return sample
    
class FixedCrop_gen(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = 10
        i1 = 10
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1]]
        
        return sample
    
class FixedCropnp(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = 0
        i1 = 0
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1],:]
        
        return sample

class RandCropnp(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = np.random.randint(0, 640-self.shape[0])
        i1 = np.random.randint(0, 1080-self.shape[1])
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1],:]
        
        #sample['rand_inds'] = [i0,i0+self.shape[0],i1,i1+self.shape[1]]
        return sample
    
class RandCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = np.random.randint(0, 640-self.shape[0])
        i1 = np.random.randint(0, 1080-self.shape[1])
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1]]
        
        #sample['rand_inds'] = [i0,i0+self.shape[0],i1,i1+self.shape[1]]
        return sample
    
class RandCrop_gen(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        i0 = np.random.randint(0, 640-self.shape[0])
        i1 = np.random.randint(0, 1080-self.shape[1])
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1]]
        
        sample['rand_inds'] = [i0,i1]
        return sample
    
class RandCrop2(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shape = (512,512)):
        self.shape = shape
    def __call__(self, sample):
        
        
        
        for key in sample:
            i0 = np.random.randint(0, sample[key].shape[-2]+1-self.shape[0])
            i1 = np.random.randint(0, sample[key].shape[-1]+1-self.shape[1])
            break
            
        for key in sample:
            sample[key] = sample[key][...,i0:i0+self.shape[0],i1:i1+self.shape[1]]
        
        return sample
    
class Get_sample_noise(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.all_files = glob.glob(input_dir + 'noisy*.mat')
        self.transform = transform

    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):

        
        sample_loaded = scipy.io.loadmat(self.all_files[idx])
        
        gt_ind = self.all_files[idx].split('noisy')[1].split('_')[0]
        gt_file = glob.glob(self.input_dir + 'gt' + gt_ind + '*.mat')
        gt_loaded = scipy.io.loadmat(gt_file[0])
        
        gt_im = gt_loaded['gt']/gt_loaded['alpha']
        
        noisy_im = sample_loaded['noisy']

        sample = {'noisy_input': noisy_im, 
              'gt_label': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample    
class Get_sample_noise_batch(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None, fixed_noise=False):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        all_files0 = glob.glob(input_dir + 'noisy0*.mat')
        all_files1 = glob.glob(input_dir + 'noisy2*.mat')
        
        unsorted_inds = [int(np.array(all_files0)[i].split('.mat')[0].split('_')[-1]) for i in range(0,len(all_files0))]
        sorted_inds = np.argsort(unsorted_inds)
        self.all_files0 = np.array(all_files0)[sorted_inds]
        
        unsorted_inds = [int(np.array(all_files1)[i].split('.mat')[0].split('_')[-1]) for i in range(0,len(all_files1))]
        sorted_inds = np.argsort(unsorted_inds)
        self.all_files1 = np.array(all_files1)[sorted_inds]
        
        self.transform = transform
        
        self.fixed_noise_opt = fixed_noise
        mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
        self.fixed_noise = mean_noise.astype('float32')/2**16

    def __len__(self):
        return len(self.all_files1) - 16
    def __getitem__(self, idx):
        
        which_one = np.random.randint(0,2)
        if which_one ==0:
            all_files = self.all_files0
        else:
            all_files = self.all_files1
        
        sample_loaded = scipy.io.loadmat(all_files[idx])
        noisy_im = np.empty((16, *sample_loaded['noisy'].shape))
        
        for i in range(0,16):
            sample_loaded = scipy.io.loadmat(all_files[idx+i])
            noisy_im[i] = sample_loaded['noisy']

        gt_ind = all_files[idx].split('noisy')[1].split('_')[0]
        gt_file = glob.glob(self.input_dir + 'gt' + gt_ind + '*.mat')
        gt_loaded = scipy.io.loadmat(gt_file[0])
        #gt_im = gt_loaded['gt']/gt_loaded['alpha']
        
        gt_im = np.empty((16, *sample_loaded['noisy'].shape))
        gt_im = np.repeat((gt_loaded['gt']/gt_loaded['alpha'])[np.newaxis], 16, axis = 0)

        if self.fixed_noise_opt == True:
            gt_im1 = gt_im + self.fixed_noise[np.newaxis]/gt_loaded['alpha']
        else:
            gt_im1 = gt_im
            
        sample = {'noisy_input': noisy_im, 
              'gt_label': gt_im1,
               'gt_label_nobias': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class Get_sample_noise_batch_new(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.input_dir = input_dir
        self.all_files = input_dir
        
        self.transform = transform
        
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        
       
        
        sample_loaded = scipy.io.loadmat(glob.glob(self.all_files[idx] +  '/clean*.mat')[0])
        gt_im = np.empty((16, *sample_loaded['gt_label'].shape))
        gt_im = np.repeat((sample_loaded['gt_label']*sample_loaded['alpha'])[np.newaxis], 16, axis = 0)
        
        all_files_noisy = glob.glob(self.all_files[idx] +  '/sequence*.mat')
        inds = []
        for i in range(0, len(all_files_noisy)):
            inds.append(int(all_files_noisy[i].split('_')[-1].split('.mat')[0]))

        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files_noisy)[inds_sort]
        idx = np.random.randint(0,len(all_files_sorted) - 17)
    
        noisy_im = np.empty_like(gt_im)
        
        for i in range(0,16):
            sample_loaded = scipy.io.loadmat(all_files_sorted[idx+i])
            noisy_im[i] = sample_loaded['gt_label']
            
        sample = {'noisy_input': (noisy_im/2**16).astype('float32'), 
               'gt_label_nobias': (gt_im/2**16).astype('float32')}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
class Get_sample_loading(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return 60
    def __getitem__(self, idx, alpha = True):
        #print(self.input_dir[idx])
        sample_loaded = scipy.io.loadmat(self.input_dir[idx])
    
        low_ind = np.random.randint(0,sample_loaded['gt_list'].shape[0])
        high_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0])

        noisy_im = sample_loaded['noisy_list'][high_ind]
        gt_im = sample_loaded['gt_list'][low_ind]

        if alpha:
            gt_im = gt_im/sample_loaded['alpha']
        else:
            noisy_im = noisy_im*sample_loaded['alpha']

        sample = {'noisy_input': noisy_im, 
              'gt_label': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class Get_sample(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return 60
    def __call__(self, im_ind, alpha = None):
        #print(self.input_dir[im_ind])
        sample_loaded = scipy.io.loadmat(self.input_dir[im_ind])
    
        low_ind = np.random.randint(0,sample_loaded['gt_list'].shape[0])
        high_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0])

        noisy_im = sample_loaded['noisy_list'][high_ind]
        gt_im = sample_loaded['gt_list'][low_ind]

        if alpha:
            gt_im = gt_im/sample_loaded['alpha']
        else:
            noisy_im = noisy_im*sample_loaded['alpha']

        sample = {'noisy_input': noisy_im, 
              'gt_label': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

    
class Get_sample_batch_fixed_noise(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        
        mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
        self.fixed_noise = mean_noise.astype('float32')/2**16
            
    def __len__(self):
        return 60
    def __getitem__(self, idx):

        alpha = True
        sample_loaded = scipy.io.loadmat(self.input_dir[idx])
    
        noisy_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]))
        gt_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]))
        gt_im = np.repeat(np.mean(sample_loaded['gt_list'], 0)[np.newaxis], 16, axis = 0)
        
        if sample_loaded['noisy_list'].shape[0]<16:
            #low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0])
            high_ind = sample_loaded['noisy_list'].shape[0]
            noisy_im[0:high_ind] = sample_loaded['noisy_list'].copy()
            noisy_im[high_ind:] = sample_loaded['noisy_list'][0:16-high_ind].copy()
        else: 
            low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0] - 16)
            noisy_im[low_ind:low_ind+16] = sample_loaded['noisy_list'][low_ind:low_ind+16].copy()

        if alpha:
            gt_im = gt_im/sample_loaded['alpha']
        else:
            noisy_im = noisy_im*sample_loaded['alpha']
        #noisy_im = noisy_im*sample_loaded['alpha']

        sample = {'noisy_input': noisy_im, 
              'gt_label': gt_im+self.fixed_noise,
                 'gt_label_real': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class Get_sample_batch(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None, start_ind = None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        self.start_ind = start_ind
    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, idx):

        alpha = True
        sample_loaded = scipy.io.loadmat(self.input_dir[idx])
    
        noisy_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]), dtype = 'float32')
        gt_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]), dtype = 'float32')
        gt_im = np.repeat(np.mean(sample_loaded['gt_list'].astype('float32'), 0)[np.newaxis], 16, axis = 0)
        
        
        if sample_loaded['noisy_list'].shape[0]<16:
            print('bad image',self.input_dir[idx])
            #low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0])
            high_ind = sample_loaded['noisy_list'].shape[0]
            noisy_im[0:high_ind] = sample_loaded['noisy_list'].copy()
            noisy_im[high_ind:] = sample_loaded['noisy_list'][0:16-high_ind].copy()
        else: 
            if self.start_ind is not None:
                low_ind = self.start_ind
            else:
                low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0] - 16)
               
            noisy_im = sample_loaded['noisy_list'].astype('float32')[low_ind:low_ind+16]#.copy()
        if alpha:
            gt_im = gt_im/sample_loaded['alpha']
        else:
            noisy_im = noisy_im*sample_loaded['alpha']
        #noisy_im = noisy_im*sample_loaded['alpha']

        sample = {'noisy_input': noisy_im, 
              #'gt_label': gt_im,
                 'gt_label_nobias': gt_im}
        
        del sample_loaded
        if self.transform:
            sample = self.transform(sample)
        
        return sample
class Get_sample_batch_video(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir = '../../Datasets/Canon/mendocino_mat/sequence_', transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, im_ind, seq_ind = None):
        
        curr_num = int(self.input_dir[im_ind].split('_')[-1].split('.mat')[0])

        all_files = glob.glob(self.input_dir[im_ind].split('sequence')[0] +'/*.mat')
        num_in_seq = len(all_files)

        inds = []
        for i in range(0, len(all_files)):
            inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))

        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files)[inds_sort]
    
        noisy_im = np.empty((16,640,1080,4))
        #if seq_ind:
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        #else:
        #    seq_ind = np.random.randint(0, num_in_seq - 16)
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        
        for i in range(0,16):
            noisy_im[i] =  scipy.io.loadmat(all_files_sorted[curr_num + i])['noisy_list'].astype('float32')    
                
        
        noisy_im = noisy_im/2**16

        sample = {'gt_label_nobias': noisy_im}
        if self.transform:
            sample = self.transform(sample)
        
        return sample  
class Get_sample_batch_video_distributed(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, generator, input_dir = '../../Datasets/Canon/mendocino_mat/sequence_', transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        self.generator = generator

    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, im_ind, seq_ind = None):
        
        curr_num = int(self.input_dir[im_ind].split('_')[-1].split('.mat')[0])

        all_files = glob.glob(self.input_dir[im_ind].split('sequence')[0] +'/*.mat')
        num_in_seq = len(all_files)

        inds = []
        for i in range(0, len(all_files)):
            inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))

        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files)[inds_sort]
    
        noisy_im = np.empty((16,640,1080,4))
        #if seq_ind:
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        #else:
        #    seq_ind = np.random.randint(0, num_in_seq - 16)
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        
        for i in range(0,16):
            noisy_im[i] =  scipy.io.loadmat(all_files_sorted[curr_num + i])['noisy_list'].astype('float32')    
                
        
        noisy_im = noisy_im/2**16

        sample = {'gt_label_nobias': noisy_im,}
        if self.transform:
            sample = self.transform(sample)
            
        with torch.no_grad():
            gt_label_t = sample['gt_label_nobias'].transpose(0,1)
            net_input = self.generator(gt_label_t.cuda(self.generator.device)).transpose(0,1)
        
        sample['noisy_input']=  net_input.to('cpu')
        
        return sample 
    
class Get_sample_batch_video_distributed2(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir = '../../Datasets/Canon/mendocino_mat/sequence_', transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, im_ind, seq_ind = None):
        
        curr_num = int(self.input_dir[im_ind].split('_')[-1].split('.mat')[0])

        all_files = glob.glob(self.input_dir[im_ind].split('sequence')[0] +'/*.mat')
        num_in_seq = len(all_files)

        inds = []
        for i in range(0, len(all_files)):
            inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))

        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files)[inds_sort]
    
        noisy_im = np.empty((16,640,1080,4))
        #if seq_ind:
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        #else:
        #    seq_ind = np.random.randint(0, num_in_seq - 16)
        #    for i in range(0,16):
        #        noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        
        for i in range(0,16):
            noisy_im[i] =  scipy.io.loadmat(all_files_sorted[curr_num + i])['noisy_list'].astype('float32')    
                
        
        noisy_im = noisy_im/2**16

        sample = {'gt_label_nobias': noisy_im,}
        if self.transform:
            sample = self.transform(sample)
            
        sample['noisy_input'] =  (sample['gt_label_nobias']*0)-5.
        
        return sample  
class Get_sample_batch_video_distributed3(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir = '../../Datasets/Canon/mendocino_mat/sequence_', transform=None, testing_ind = None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        self.testing_ind = testing_ind

    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, im_ind, seq_ind = None):
        
        all_files = glob.glob(self.input_dir[im_ind] + '/*.mat')
        inds = []
        for i in range(0, len(all_files)):
            inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))
        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files)[inds_sort]

        print(len(all_files_sorted), self.input_dir[im_ind])
        if seq_ind is None:
            seq_ind = np.random.randint(0,len(all_files_sorted)-16)
        if self.testing_ind is not None:
            seq_ind = self.testing_ind

        noisy_im = np.empty((16,640,1080,4))
        for i in range(0,16):     
            loaded_mat = scipy.io.loadmat(all_files_sorted[seq_ind + i])

            if 'gt_label' in loaded_mat:
                noisy_im[i] =  loaded_mat['gt_label'].astype('float32')    
            elif 'noisy_list' in loaded_mat:
                noisy_im[i] =  loaded_mat['noisy_list'].astype('float32')    
            else:
                print('invalid loaded file')


        print('mas: ', np.max(noisy_im))
        noisy_im = noisy_im/(2**16)

        sample = {'gt_label_nobias': noisy_im,}
        if self.transform:
            sample = self.transform(sample)
            
        sample['noisy_input'] =  (sample['gt_label_nobias']*0)-5.
        
        return sample  
class Get_sample_batch_unsupervised(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir = '../../Datasets/Canon/mendocino_mat/sequence_', transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return 8
    def __getitem__(self, im_ind, seq_ind = None):

        #all_files = glob.glob('../../Datasets/Canon/mendocino_mat/sequence_' +str(im_ind) +'*.mat')
        all_files = glob.glob(self.input_dir +str(im_ind) +'*.mat')
        num_in_seq = len(all_files)
        print(im_ind, num_in_seq, all_files[0])
        print(scipy.io.loadmat(all_files[0])['name'])
        
        inds = []
        for i in range(0, len(all_files)):
            inds.append(int(all_files[i].split('_')[-1].split('.mat')[0]))

        inds_sort = np.argsort(inds)
        all_files_sorted = np.array(all_files)[inds_sort]
    
        noisy_im = np.empty((16,640,1080,4))
        if seq_ind:
            for i in range(0,16):
                print('here', seq_ind+i, scipy.io.loadmat(all_files_sorted[seq_ind+i])['name'])
                noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
        else:
            seq_ind = np.random.randint(0, num_in_seq - 16)
            for i in range(0,16):
                noisy_im[i] =  scipy.io.loadmat(all_files_sorted[seq_ind+i])['noisy_list'].astype('float32')
                
                
        noisy_im = noisy_im/2**16

        sample = {'noisy_input': noisy_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats

def gen_noise(im):
    shape = im.shape
    loaded_row = scipy.io.loadmat('/home/kmonakho/Datasets/Canon/bias_rownoise.mat')
    loaded_read = scipy.io.loadmat('/home/kmonakho/Datasets/Canon/bias_readnoise.mat')
    
    dist_Nr0 = getattr(scipy.stats, loaded_row['Nr_0_type'][0])
    dist_Nr1 = getattr(scipy.stats, loaded_row['Nr_1_type'][0])
    dist_Nr2 = getattr(scipy.stats, loaded_row['Nr_2_type'][0])
    dist_Nr3 = getattr(scipy.stats, loaded_row['Nr_3_type'][0])

    Nr_0_param = loaded_row['Nr_0_param'][0]
    Nr_1_param = loaded_row['Nr_1_param'][0]
    Nr_2_param = loaded_row['Nr_2_param'][0]
    Nr_3_param = loaded_row['Nr_3_param'][0]

    if len(shape) ==5:
        shape_new = [shape[0], shape[2], 1, shape[-1]]
    else:
        shape_new = [shape[0], 1, shape[2]]
    my_gen_noise0 = dist_Nr0.rvs(loc = .25*Nr_0_param[0], scale =  .25*Nr_0_param[1], size=np.prod(shape_new)).reshape(shape_new)
    my_gen_noise1 = dist_Nr1.rvs(loc = .25*Nr_1_param[0], scale =  .25*Nr_1_param[1], size=np.prod(shape_new)).reshape(shape_new)
    my_gen_noise2 = dist_Nr2.rvs(loc = .25*Nr_2_param[0], scale =  .25*Nr_2_param[1], size=np.prod(shape_new)).reshape(shape_new)
    my_gen_noise3 = dist_Nr3.rvs(loc = .25*Nr_3_param[0], scale =  .25*Nr_3_param[1], size=np.prod(shape_new)).reshape(shape_new)

    my_gen_image = np.stack([my_gen_noise0, my_gen_noise1, my_gen_noise2, my_gen_noise3], -1)
    
    dist_read = getattr(scipy.stats, loaded_read['Nread_type'][0])
    read_param = loaded_read['Nread_param'][0]
    
    my_gen_read = dist_read.rvs(*read_param*.25, size=np.prod(shape)).reshape(shape)

    dist_quant = getattr(scipy.stats, 'uniform')
    my_gen_quant = dist_quant.rvs(loc = -1/(2), scale =2*1/(2), size=np.prod(shape)).reshape(shape)
    
    final_image = np.clip(im + np.random.randn(*shape)*0.024*im + my_gen_read+my_gen_image+my_gen_quant, 0, 2**14)
    final_image = np.float32(np.uint16(final_image))
    return final_image/2**14

class Get_sample_batch_simulated(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

    def __call__(self, im_ind, alpha = None):

        sample_loaded = scipy.io.loadmat(self.input_dir[im_ind])
    
        noisy_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]))
        gt_im = np.empty((16, *sample_loaded['noisy_list'].shape[1:]))
        gt_im = np.repeat(np.mean(sample_loaded['gt_list'], 0)[np.newaxis], 16, axis = 0)
        
        if sample_loaded['noisy_list'].shape[0]<16:
            #low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0])
            high_ind = sample_loaded['noisy_list'].shape[0]
            noisy_im[0:high_ind] = sample_loaded['noisy_list'].copy()
            noisy_im[high_ind:] = sample_loaded['noisy_list'][0:16-high_ind].copy()
        else: 
            low_ind = np.random.randint(0,sample_loaded['noisy_list'].shape[0] - 16)
            noisy_im[low_ind:low_ind+16] = sample_loaded['noisy_list'][low_ind:low_ind+16].copy()

        if alpha:
            gt_im = gt_im/sample_loaded['alpha']
        else:
            noisy_im = noisy_im*sample_loaded['alpha']
            
        gen_noisy = gen_noise(gt_im*2**14)
        #noisy_im = noisy_im*sample_loaded['alpha']

        sample = {'noisy_input': gen_noisy, 
                  'noisy_input_real': noisy_im, 
              'gt_label': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
class Get_sample_batch_simvideo(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None, start_ind = None):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        
        mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
        self.fixed_noise = mean_noise.astype('float32')/2**16
        
        self.start_ind = start_ind
            
    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, idx):

        sorted_ims = np.sort(glob.glob(self.input_dir[idx]+'/*'))
        
        if self.start_ind is not None:
                low_ind = self.start_ind
        else:
            low_ind = np.random.randint(0,len(sorted_ims) - 15)
        
        im_clean = np.array(Image.open(sorted_ims[low_ind]), dtype='float32')/2**12
        #print(idx, 'clean loaded', im_clean.shape)
        im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                          im_clean[1::2,1::2],
                          im_clean[0::2,0::2], 
                          im_clean[0::2,1::2], 
                          ], -1)
        
        
        
        gt_im = np.empty((16, *im_clean_rgbg.shape))
        gt_im[0] = im_clean_rgbg
        for i in range(1,16):
            im_clean = np.array(Image.open(sorted_ims[low_ind+i]), dtype='float32')/2**12
            im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                              im_clean[1::2,1::2],
                              im_clean[0::2,0::2], 
                              im_clean[0::2,1::2], 
                              ], -1)
            
            gt_im[i] = im_clean_rgbg
        
        #noisy_im = noisy_im*sample_loaded['alpha']
        
        if gt_im.shape[1]>gt_im.shape[2]:
            gt_im = gt_im.transpose(0,2,1,3)

        sample = {'gt_label_nobias': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class Get_sample_batch_simvideo_distributed(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, generator, input_dir, transform=None, start_ind = None, crop_size = (512,512)):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        
        mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
        self.fixed_noise = mean_noise.astype('float32')/2**16
        
        self.start_ind = start_ind
        
        self.generator = generator
        self.crop_size = crop_size
            
    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, idx):

        sorted_ims = np.sort(glob.glob(self.input_dir[idx]+'/*'))
        
        if self.start_ind is not None:
                low_ind = self.start_ind
        else:
            low_ind = np.random.randint(0,len(sorted_ims) - 15)
        
        im_clean = np.array(Image.open(sorted_ims[low_ind]), dtype='float32')/2**12
        #print(idx, 'clean loaded', im_clean.shape)
        im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                          im_clean[1::2,1::2],
                          im_clean[0::2,0::2], 
                          im_clean[0::2,1::2], 
                          ], -1)
        
        
        if im_clean_rgbg.shape[0]<self.crop_size[0] or im_clean_rgbg.shape[1]<self.crop_size[1]:
            new_size0 = np.maximum(im_clean_rgbg.shape[0],self.crop_size[0])
            new_size1 = np.maximum(im_clean_rgbg.shape[1],self.crop_size[1])
            im_clean_rgbg = cv2.resize(im_clean_rgbg, (new_size0, new_size1))
            upsample_image = True
        else:
            upsample_image = False
            
        gt_im = np.empty((16, *im_clean_rgbg.shape))
        gt_im[0] = im_clean_rgbg
        for i in range(1,16):
            im_clean = np.array(Image.open(sorted_ims[low_ind+i]), dtype='float32')/2**12
            im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                              im_clean[1::2,1::2],
                              im_clean[0::2,0::2], 
                              im_clean[0::2,1::2], 
                              ], -1)
            
            if upsample_image == True:
                im_clean_rgbg = cv2.resize(im_clean_rgbg, (new_size0, new_size1))
                
            gt_im[i] = im_clean_rgbg
        
        #noisy_im = noisy_im*sample_loaded['alpha']
        
        if gt_im.shape[1]>gt_im.shape[2]:
            gt_im = gt_im.transpose(0,2,1,3)

        sample = {'gt_label_nobias': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
            
        with torch.no_grad():
            gt_label_t = sample['gt_label_nobias'].transpose(0,1)
            net_input = self.generator(gt_label_t.cuda(self.generator.device)).transpose(0,1)

        sample['noisy_input']=  net_input.to('cpu')
        
        return sample
class Get_sample_batch_simvideo_distributed2(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, input_dir, transform=None, start_ind = None, crop_size = (512,512)):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        
        mean_noise = scipy.io.loadmat('data/fixed_pattern_noise.mat')['mean_pattern']
        self.fixed_noise = mean_noise.astype('float32')/2**16
        
        self.start_ind = start_ind

        self.crop_size = crop_size
            
    def __len__(self):
        return len(self.input_dir)
    def __getitem__(self, idx):

        sorted_ims = np.sort(glob.glob(self.input_dir[idx]+'/*'))
        
        if self.start_ind is not None:
                low_ind = self.start_ind
        else:
            low_ind = np.random.randint(0,len(sorted_ims) - 15)
        
        im_clean = np.array(Image.open(sorted_ims[low_ind]), dtype='float32')/2**12
        #print(idx, 'clean loaded', im_clean.shape)
        im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                          im_clean[1::2,1::2],
                          im_clean[0::2,0::2], 
                          im_clean[0::2,1::2], 
                          ], -1)
        
        
        if im_clean_rgbg.shape[0]<self.crop_size[0] or im_clean_rgbg.shape[1]<self.crop_size[1]:
            new_size0 = np.maximum(im_clean_rgbg.shape[0],self.crop_size[0])
            new_size1 = np.maximum(im_clean_rgbg.shape[1],self.crop_size[1])
            im_clean_rgbg = cv2.resize(im_clean_rgbg, (new_size0, new_size1))
            upsample_image = True
        else:
            upsample_image = False
            
        gt_im = np.empty((16, *im_clean_rgbg.shape))
        gt_im[0] = im_clean_rgbg
        for i in range(1,16):
            im_clean = np.array(Image.open(sorted_ims[low_ind+i]), dtype='float32')/2**12
            im_clean_rgbg = np.stack([im_clean[1::2,0::2], 
                              im_clean[1::2,1::2],
                              im_clean[0::2,0::2], 
                              im_clean[0::2,1::2], 
                              ], -1)
            
            if upsample_image == True:
                im_clean_rgbg = cv2.resize(im_clean_rgbg, (new_size0, new_size1))
                
            gt_im[i] = im_clean_rgbg
        
        
        if gt_im.shape[1]>gt_im.shape[2]:
            gt_im = gt_im.transpose(0,2,1,3)

        sample = {'gt_label_nobias': gt_im}
        
        if self.transform:
            sample = self.transform(sample)
            
        
        sample['noisy_input'] =  (sample['gt_label_nobias']*0)-5.
        
        return sample    
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)