import cv2
import numpy as np
import os
import sys
import mimetypes
from mrcnn import utils
from mrcnn import model as modellib

ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
sys.path.append(os.path.join(ROOT_DIR,"samples/coco/"))
import coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def background_options(option):
    bg_options =  {'green' : [0,255,0],
                   'blue' : [245,39,8]}
    return bg_options[option]

def create_background(image, option, vbg):
    if option == 'gray':
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #bg_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bg_image = np.zeros_like(image)
        for i in range(3):
            bg_image[:,:,i] = gray_image
    elif option == 'external':
        if vbg is None:
            print('no virtual background detected')
            sys.exit()   
        bg_image = cv2.resize(vbg,(image.shape[1], image.shape[0]))
    elif option == 'blue' or option =='green':
        bg_image = np.zeros_like(image)
        bg_image[:] = background_options(option)  
    else:
        print('Unidentified Mode. Please choose between blue, green, external, or gray')
    return bg_image    

def apply_mask(image, mask,background,vbg):
    bg_image = create_background(image, background,vbg)
    for i in range(3):
        image[:, :, i] = np.where(
            mask == 0,
            bg_image[:, :, i],
            image[:, :, i]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores, bg_option,vbg=None):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    max_bbox_size = 0
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i] #extract coordinates of box
        bbox_area = (y2-y1) * (x2-x1)
        label = names[ids[i]] #get class name from id
        print(label)
        if label =='person':
            if bbox_area > max_bbox_size:
                max_bbox_size = bbox_area
                mask = masks[:, :, i]
            else:
                continue 
        else:
            continue
        image = apply_mask(image, mask,bg_option,vbg)

    return image


