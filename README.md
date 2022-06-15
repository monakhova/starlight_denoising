# Dancing under the stars: video denoising in starlight

### [Project Page](http://kristinamonakhova.com/starlight_denoising/) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Monakhova_Dancing_Under_the_Stars_Video_Denoising_in_Starlight_CVPR_2022_paper.html) | [Dataset](http://kristinamonakhova.com/starlight_denoising/#dataset)


### Setup:
Clone this project using:
```
git clone https://github.com/monakhova/starlight_denoising
```

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate starlight
```

### Loading in the pre-trained models
Our pre-trained noise model and denoiser model can be downloaded [here](https://drive.google.com/drive/folders/1Tf3R6MqSlzfPXExkbDP7FjPhU1Ak4p43?usp=sharing).


### Dataset 
Instructions for downloading our full dataset can be found [here](https://kristinamonakhova.com/starlight_denoising/#dataset). For denoiser demonstration purposes, we also provide a curated, smaller dataset (1.3GB) [here](https://drive.google.com/drive/folders/1ztbuJElSdT2MTOm1RgGnSEDFXIsBHO5q?usp=sharing). This can be used for our denoising demo notebook. 

### Dataset and denoising demo
We provide a [jupyter notebook demo](https://github.com/monakhova/starlight_denoising/blob/main/Denoise%20Submillilux%20Videos.ipynb) to showcase our dataset and pretrained denoiser performance. Please download our pre-trained models and our full or condensed dataset to view this demo. In this demo, we show our raw noisy submillilux video clips and then our denoiser performance. 

### Noise generator demo
We provide a [jupyter notebook demo](https://github.com/monakhova/starlight_denoising/blob/main/View%20Generated%20Noise.ipynb) to view our learned noise model and showcase our generated noisy clips. To view this demo, please download our paired data, which you can find [here](https://drive.google.com/drive/folders/1xIxUfzkSf1pmCgY3QnrYTorc9ZoUlx7w?usp=sharing) and place it in data/paired_data/. 


### Noise generator training code
Coming soon.

### Denoiser training code
Coming soon.


### Citation
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