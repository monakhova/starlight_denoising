
def load_3d_hrnet(num_channels=4):
    from models.seg_hrnet3d import HighResolutionNet
    from yacs.config import CfgNode as CN
    
    _C= CN()

    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.ALIGN_CORNERS = True
    _C.MODEL.NUM_INPUT_CHANNELS = num_channels
    _C.MODEL.NUM_OUTPUTS = 1
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.MODEL.EXTRA.FINAL_CONV_KERNEL= 1
    _C.MODEL.EXTRA.STAGE1 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
    _C.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
    _C.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
    _C.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
    _C.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE2 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
    _C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4,4]
    _C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE3 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
    _C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4,4,4]
    _C.MODEL.EXTRA.STAGE3.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE4 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
    _C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4,4,4,4]
    _C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'


    _C.DATASET = CN()
    _C.DATASET.NUM_CLASSES = num_channels

    cfg = _C
    model_hr = HighResolutionNet(cfg)
    
    return model_hr

def load_3d_hrnet_small(num_channels=4):
    from models.seg_hrnet3d import HighResolutionNet
    from yacs.config import CfgNode as CN
    
    _C= CN()

    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.ALIGN_CORNERS = True
    _C.MODEL.NUM_INPUT_CHANNELS = num_channels
    _C.MODEL.NUM_OUTPUTS = 1
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.MODEL.EXTRA.FINAL_CONV_KERNEL= 1
    _C.MODEL.EXTRA.STAGE1 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
    _C.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
    _C.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
    _C.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
    _C.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE2 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [8, 16]
    _C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4,4]
    _C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE3 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [8, 16, 32]
    _C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4,4,4]
    _C.MODEL.EXTRA.STAGE3.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE4 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [8, 16, 32, 64]
    _C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4,4,4,4]
    _C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'


    _C.DATASET = CN()
    _C.DATASET.NUM_CLASSES = num_channels

    cfg = _C
    model_hr = HighResolutionNet(cfg)
    
    return model_hr

def load_2d_hrnet(num_channels=64):
    from models.seg_hrnet import HighResolutionNet
    from yacs.config import CfgNode as CN
    
    _C= CN()

    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.ALIGN_CORNERS = True
    _C.MODEL.NUM_INPUT_CHANNELS = num_channels
    _C.MODEL.NUM_OUTPUTS = 1
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.MODEL.EXTRA.FINAL_CONV_KERNEL= 1
    _C.MODEL.EXTRA.STAGE1 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
    _C.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
    _C.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
    _C.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
    _C.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE2 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
    _C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4,4]
    _C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE3 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
    _C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4,4,4]
    _C.MODEL.EXTRA.STAGE3.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE4 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
    _C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4,4,4,4]
    _C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'


    _C.DATASET = CN()
    _C.DATASET.NUM_CLASSES = num_channels

    cfg = _C
    model_hr = HighResolutionNet(cfg)
    
    return model_hr

def load_2d_hrnet2(num_channels=64, num_classes = 4):
    from models.seg_hrnet import HighResolutionNet
    from yacs.config import CfgNode as CN
    
    _C= CN()

    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.ALIGN_CORNERS = True
    _C.MODEL.NUM_INPUT_CHANNELS = num_channels
    _C.MODEL.NUM_OUTPUTS = 1
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.MODEL.EXTRA.FINAL_CONV_KERNEL= 1
    _C.MODEL.EXTRA.STAGE1 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE1.NUM_CHANNELS = [64]
    _C.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
    _C.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
    _C.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
    _C.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE2 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
    _C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4,4]
    _C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    _C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE3 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
    _C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4,4,4]
    _C.MODEL.EXTRA.STAGE3.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    _C.MODEL.EXTRA.STAGE4 =  CN(new_allowed=True)
    _C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
    _C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    _C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4,4,4,4]
    _C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    _C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'


    _C.DATASET = CN()
    _C.DATASET.NUM_CLASSES = num_classes

    cfg = _C
    model_hr = HighResolutionNet(cfg)
    
    return model_hr