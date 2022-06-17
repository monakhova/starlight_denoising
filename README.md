# Dancing under the stars: video denoising in starlight

## [Project Page](http://kristinamonakhova.com/starlight_denoising/) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Monakhova_Dancing_Under_the_Stars_Video_Denoising_in_Starlight_CVPR_2022_paper.html) | [Dataset](http://kristinamonakhova.com/starlight_denoising/#dataset)


## Setup:
Clone this project using:
```
git clone https://github.com/monakhova/starlight_denoising
```

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate starlight
```

## Loading in the pre-trained models
Our pre-trained noise model and denoiser model can be downloaded [here](https://drive.google.com/drive/folders/1Tf3R6MqSlzfPXExkbDP7FjPhU1Ak4p43?usp=sharing).


## Dataset 
Instructions for downloading our full dataset can be found [here](https://kristinamonakhova.com/starlight_denoising/#dataset). For denoiser demonstration purposes, we also provide a curated, smaller dataset (1.3GB) [here](https://drive.google.com/drive/folders/1ztbuJElSdT2MTOm1RgGnSEDFXIsBHO5q?usp=sharing). This can be used for our denoising demo notebook. 

We save and access images from our dataset as .mat files for easy loading/accessing. If you'd like to use our raw .dng files, you can use the following code snippet to load in the images:

```
from canon_utils import read_16bit_raw, raw_to_4
raw_image = raw_to_4(read_16bit_raw('image_name.dng').astype('uint16')
```
This will read in the RAW file format, then split the image into 4 channels (R,G,B,NIR). Note that these functions do not perform demosaicing (we simply split up the color channels), so the resulting image will be a half the size of the original RAW image.

## Dataset and denoising demo
We provide a [jupyter notebook demo](https://github.com/monakhova/starlight_denoising/blob/main/Denoise%20Submillilux%20Videos.ipynb) to showcase our dataset and pretrained denoiser performance. Please download our pre-trained models and our full or condensed dataset to view this demo. In this demo, we show our raw noisy submillilux video clips and then our denoiser performance. 

## Noise generator demo
We provide a [jupyter notebook demo](https://github.com/monakhova/starlight_denoising/blob/main/View%20Generated%20Noise.ipynb) to view our learned noise model and showcase our generated noisy clips. To view this demo, please download our paired data, which you can find [here](https://drive.google.com/drive/folders/1xIxUfzkSf1pmCgY3QnrYTorc9ZoUlx7w?usp=sharing) and place it in data/paired_data/. 


## Noise generator training code
Our code to train our noise generator can be found [here](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_gan_noisemodel.py). This code takes in a small dataset of paired clean/noisy image bursts and learns a physics-informed noise model to represent realistic noise for a camera at a fixed gain setting. To train our GAN noise model using our data, please download our paired training dataset which can be found [here](https://drive.google.com/drive/folders/1xIxUfzkSf1pmCgY3QnrYTorc9ZoUlx7w?usp=sharing). Please place this data within the data folder. 

To run the training script, run the following command:
```
python train_gan_noisemodel.py --batch_size 1 --gpus 8 --noiselist shot_read_uniform_row1_rowt_fixed1_periodic
```

We include options to change the physics-inspired noise parameters. For example, if you only want to include read and shot noise, you can run the script with the options ```--noiselist shot_read```. In addition, we provide options for including or excluding the U-Net from the noise model, options for specifying the dataset, and changing the discriminator loss to operate in real or fourier space.

### Check noise generator performance after training
To check out noise generator performance, run our noise generator jupyter notebook demo, [View Generated Noise.ipynb](https://github.com/monakhova/starlight_denoising/blob/main/View%20Generated%20Noise.ipynb), and change the following line to go to your saved checkpoint:

```
chkp_path = 'checkpoint_name_here'
```

By default, checkpoints are saved in /saved_models/ 


### Use noise generator for denoiser training
To use your noise model during denoiser training, update the function [load_generator_model](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_denoiser.py#L364) in train_denoiser.py to include your noise model name and path:
```
elif args.noise_type == 'your_noise_model_here': 
        base_file = '../saved_models/your_noise_model_path_here/'
```
Then, rerun train_denoiser.py with the argument ```--noise_type your_noise_model_here```

### Running on SLURM cluster
In addition, we provide an example script to run this code on a SLURM cluster [here](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_noise_script.sh). You can run this using:
```
sbatch train_noise_script.sh 
```

## Denoiser training code
Our code to retrain the video denoiser from scratch can be found [here](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_denoiser.py). By default, we use a crop size of 512x512 and run the training on 16 GPUs. You can adjust the number of GPUs, the crop size, and many other paramters, such as the save path, dataset path, the dataset type, and the network architecture through the run options. For example, we can run the script as:

```
python train_denoiser.py --batch_size 1 --gpus 8 --notes default_settings --crop_size 512 --data stills_realvideo --network dvdhr
```

Please see [scripts/train_denoiser.py](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_denoiser.py) for full options and information. If you run out of memory, scale down the ```crop_size``` until it fits on your GPU (options: 'small', '256', '512', and 'full'). Note that during our experiments, we used 48GB GPUs. We also provide options for preloading a pre-trained model or resuming training, as well as options to change the noise model. 

Note that we pretrain our video denoiser using unprocessed videos from a subset of the MOT dataset. To include this in your traininig, please download the [MOT dataset](https://motchallenge.net/) and [unprocess](https://github.com/timothybrooks/unprocessing) the images. Once you have done this, you can update the filepath to your unprocessed MOT images, ```--MOT_path```, and add MOTvideo to ```--data```. 

### Running on SLURM cluster
In addition, we provide an example script to run this code on a SLURM cluster [here](https://github.com/monakhova/starlight_denoising/blob/main/scripts/train_denoiser_script.sh). You can run this using:
```
sbatch train_denoiser_script.sh 
```

### Check denoiser performance after training
To check out the denoiser performance on our submillilux dataset, run our denoiser jupyter notebook demo, [Denoise Submillilux Videos.ipynb](https://github.com/monakhova/starlight_denoising/blob/main/Denoise%20Submillilux%20Videos.ipynb), and change the following line to go to your saved checkpoint:

```
chkp_path = 'checkpoint_name_here'
```

By default, checkpoints are saved in /saved_models/ 

## Citation
Please cite this work as:
```
@InProceedings{Monakhova_2022_CVPR,
    author    = {Monakhova, Kristina and Richter, Stephan R. and Waller, Laura and Koltun, Vladlen},
    title     = {Dancing Under the Stars: Video Denoising in Starlight},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16241-16251}
}
```