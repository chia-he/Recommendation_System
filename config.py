# =============================================================================
# GazeFollow dataset dir config
# =============================================================================
'''
gazefollow_train_data = "data/gazefollow"
gazefollow_train_label = "data/gazefollow/train_annotations_release.txt"
gazefollow_val_data = "data/gazefollow"
gazefollow_val_label = "data/gazefollow/test_annotations_release.txt"
'''

# =============================================================================
# VideoAttTarget dataset dir config
# =============================================================================
'''
videoattentiontarget_train_data = "data/videoatttarget/images"
videoattentiontarget_train_label = "data/videoatttarget/annotations/train"
videoattentiontarget_val_data = "data/videoatttarget/images"
videoattentiontarget_val_label = "data/videoatttarget/annotations/test"
'''

# =============================================================================
# model config
# =============================================================================
input_resolution = 224 
output_resolution = 64 
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''
RESOLUTION ={'4:3':[(320,240),(640,480),(800,600),(1024,768),(1152,864)],
            '16:9':[(640,360),(960,540),(1280,720),(1366,768),(1920,1080)],
            'HD':(1280,720),'FHD':(1920,1080),'2K':(2048,1080),
            'QHD':(2560,1080),'4K':(3840,2160),'8K':(7680,4320)
            } 
'''