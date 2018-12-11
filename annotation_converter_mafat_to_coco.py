# zvika - change of Kfir's origin script to convert from MAFAT DB to msCOCO.
# kfir akons 11.10.18 # this script creates annotations DB for openImages in the format of msCOCO.

import json
import os
import pandas as pd
import time
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

#Train:
# src_csv_file = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/annotations/train_extended.csv'
# dst_file_path = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/annotations/instances_training_imagery.json'
# images_path = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/training_imagery'

#Val:
src_csv_file = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/annotations/train_extended.csv'
dst_file_path = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/annotations/instances_val_imagery.json'
images_path = '/media/Data/Data/Mafat_Challenge/Dataset_for_participants_V2/val_imagery'


class _MafatMeta(object):
    INSTANCE_TO_BASEDIR = {
        'training_imagery': 'training_imagery',
    }

    def valid(self):
        return hasattr(self, 'cat_names')

    def create(self, cat_ids, cat_names):
        """
        cat_ids: list of ids
        cat_names: list of names
        """
        assert not self.valid()
        #assert len(cat_ids) == cfg.DATA.NUM_CATEGORY and len(cat_names) == cfg.DATA.NUM_CATEGORY
        self.cat_names = cat_names
        self.class_names = ['BG'] + self.cat_names

        # background has class id of 0
        self.category_id_to_class_id = {
            v: i + 1 for i, v in enumerate(cat_ids)}
        self.class_id_to_category_id = {
            v: k for k, v in self.category_id_to_class_id.items()}
        #cfg.DATA.CLASS_NAMES = self.class_names

MafatMeta = _MafatMeta()


SIZE_DICT = 0
##################################################

import struct

def list_images_containing_category(catName):
    # ['sedan', 'truck', 'dedicated agricultural vehicle', 'jeep', 'crane truck', 'prime mover',
    # 'cement mixer', 'hatchback', 'minivan', 'pickup', 'van', 'light truck', 'bus', 'tanker', 'minibus']

    # initializing coco object to read, search, and visualize COCO dataset
    mafat_in_coco_fmt = COCO(annotation_file=dst_file_path)

    # initialize the meta
    cat_ids = mafat_in_coco_fmt.getCatIds()
    cat_names = [c['name'] for c in mafat_in_coco_fmt.loadCats(cat_ids)]
    cat_dict = dict(zip(cat_ids, cat_names))
    assert [catName] in cat_dict.values()

    catId = list(cat_dict.keys())[list(cat_dict.values()).index([catName])]
    img_ids_which_contain_catName = sorted(mafat_in_coco_fmt.getImgIds(catIds=catId))
    print(img_ids_which_contain_catName)


def viz_random_annotation(catName):
    #['sedan', 'truck', 'dedicated agricultural vehicle', 'jeep', 'crane truck', 'prime mover',
    # 'cement mixer', 'hatchback', 'minivan', 'pickup', 'van', 'light truck', 'bus', 'tanker', 'minibus']

    # initializing coco object to read, search, and visualize COCO dataset
    mafat_in_coco_fmt = COCO(annotation_file=dst_file_path)

    # initialize the meta
    cat_ids = mafat_in_coco_fmt.getCatIds()
    cat_names = [c['name'] for c in mafat_in_coco_fmt.loadCats(cat_ids)]
    cat_dict = dict(zip(cat_ids,cat_names))
    assert [catName] in cat_dict.values()

    catId = list(cat_dict.keys())[list(cat_dict.values()).index([catName])]
    img_ids_which_contain_catName = mafat_in_coco_fmt.getImgIds(catIds=catId)
    imgId = img_ids_which_contain_catName[np.random.randint(0,len(img_ids_which_contain_catName))] #zvika - can change to specific id
    img = mafat_in_coco_fmt.loadImgs(imgId)[0]

    I = matplotlib.image.imread(images_path+'/'+img['file_name'])
    #plt.imshow(I); plt.axis('off')
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(I)
    annIds = mafat_in_coco_fmt.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
    anns = mafat_in_coco_fmt.loadAnns(annIds)
    # Create a Rectangle patch
    for ind in range(len(anns)):
        rect = patches.Rectangle((anns[ind]['bbox'][0],anns[ind]['bbox'][1]),anns[ind]['bbox'][2],anns[ind]['bbox'][3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Add the patch to the Axes
    plt.show()


####################
def get_im_size(fname):
    with open(fname, 'rb') as fh:
        head = fh.read(24)
        if len(head) != 24:
            return
        fh.seek(0)
        size = 2
        ftype = 0
        while not 0xc0 <= ftype <= 0xcf:
            fh.seek(size, 1)
            byte = fh.read(1)
            while ord(byte) == 0xff:
                byte = fh.read(1)
            ftype = ord(byte)
            size = struct.unpack('>H', fh.read(2))[0] - 2
        fh.seek(1, 1)
        h, w = struct.unpack('>HH', fh.read(4))

        return w, h


def rm_nonexist_ids(db):
    # removes non existent ids that exist in annotations but not in images (because of use of size_dict)

    IDS = [a['id'] for a in db['images']]
    IDS.sort()

    num_IDS = len(IDS)
    print('num of annos=' + str(len(db['annotations'])))
    print('num of valid images=' + str(num_IDS))

    new_anns = []

    i_im = 0
    i_ann = 0
    anns_removed = []
    image_id_popped = []
    for ann in db['annotations']:
        if i_ann % 500000 == 0:
            print(i_ann)
        if ann['id'] == IDS[i_im]:
            new_anns.append(ann)
        else:
            i_im += 1
            # if i_im==num_IDS:
            #   print('reached len of IDS')
            #   break
            if ann['id'] == IDS[i_im]:
                new_anns.append(ann)
            else:
                print('removed anno #' + str(i_ann))
                anns_removed.append(i_ann)
                image_id_popped.append(ann['id'])
                i_im -= 1
                # if len(anns_removed)==5: break
        i_ann += 1

    print('len of new anns=' + str(len(new_anns)))

    db['annotations'] = new_anns

    return db


def is_image_exists_in_dir(images=[],images_path='/', img_id=0):
    file_exists = False  # init
    filepath = '/'
    filename = '/'
    if os.path.isfile(os.path.join(images_path, str(img_id) + '.jpg')):
        filename = str(img_id) + '.jpg'
        filepath = os.path.join(images_path, filename)
        file_exists = True
    elif os.path.isfile(os.path.join(images_path, str(img_id) + '.tif')):
        filename = str(img_id) + '.tif'
        filepath = os.path.join(images_path, filename)
        file_exists = True
    elif os.path.isfile(os.path.join(images_path, str(img_id) + '.tiff')):
        filename = str(img_id) + '.tiff'
        filepath = os.path.join(images_path, filename)
        file_exists = True

    return (file_exists, filepath, filename)


##################################################
##################################################
##################################################
if __name__ == "__main__":
    if SIZE_DICT:
        with open(size_list_path, 'r') as f:
            size_dict = json.load(f)

    t = time.time()
    mafat_anns = pd.read_csv(src_csv_file)

    # create categories list of dicts
    #################################
    cls_table = pd.read_csv(src_csv_file, usecols=["sub_class", "general_class"])
    sub_class_list = pd.read_csv(src_csv_file, usecols=["sub_class"]).values.tolist()
    gen_class_list = pd.read_csv(src_csv_file, usecols=["general_class"]).values.tolist()

    cat_ind = 1
    categories = []
    for sub_class_idx, name in enumerate(sub_class_list):
        # add to categories only if we didnt add it before
        if not any(d['name'] == name for d in categories):
            categories.append({'supercategory': gen_class_list[sub_class_idx], 'id': cat_ind, 'name': name})
            cat_ind+=1
    print('Created {} categories'.format(len(categories)))

    # create images list of dicts
    #################################
    images = []
    imageIDs = mafat_anns.as_matrix(['image_id']).squeeze()
    imageIDsGroupedList = []

    for imageID in imageIDs:
        # add to images list only if we didn't add it before
        if not any(d == imageID for d in imageIDsGroupedList):
            imageIDsGroupedList.append(imageID)


    for id in set(imageIDsGroupedList):
        (file_exists, filepath, filename) = is_image_exists_in_dir(images, images_path, id)
        if file_exists:
            w, h = get_im_size(filepath) #works only for jpgs
            w = 900  # Bad - need to change to a working get_im_size()
            h = 600
            images.append({'id': int(id), 'file_name': filename, 'width': w, 'height': h})

    print('Created {} images'.format(len(images)))

    # create annotations list of dicts
    tagIDs = mafat_anns.as_matrix(['tag_id']).squeeze() #Mafat unique numbers - should be 11617
    imageIDs = mafat_anns.as_matrix(['image_id']).squeeze()
    subClassLabels = mafat_anns.as_matrix(['sub_class']).squeeze()
    bboxes = mafat_anns.as_matrix(['p1_x', ' p2_x', ' p3_x', ' p4_x', 'p_1y', ' p2_y', ' p3_y', ' p4_y']).tolist() #Mafat bboxes - should be 11617
    assert len(tagIDs) == len(imageIDs) == len(subClassLabels) == len(bboxes)
    annotations = []
    i = 0
    for tagID, imageId, subClassLabel, bbox in zip(tagIDs, imageIDs, subClassLabels, bboxes):

        (file_exists, filepath, filename) = is_image_exists_in_dir(images, images_path, imageId)
        if file_exists:
            # sub_class_idx = [item for item in categories if item["name"] == [subClassLabel]][0]['id'] - zvika - need to check
            sub_class_idx = next(item for item in categories if item["name"] == [subClassLabel])['id']
            x_min = min(bbox[0],bbox[1],bbox[2],bbox[3])
            x_max = max(bbox[0],bbox[1],bbox[2],bbox[3])
            y_min = min(bbox[4],bbox[5],bbox[6],bbox[7])
            y_max = max(bbox[4],bbox[5],bbox[6],bbox[7])
            annotations.append({'image_id': int(imageId), 'id': int(tagID), 'category_id': sub_class_idx, 'bbox': [x_min,y_min,(x_max-x_min),(y_max-y_min)], 'iscrowd': 0})
            i += 1

    print('Created {} annotations'.format(len(annotations)))

     # build db
    db = {'images': images,
          'annotations': annotations,
          'categories': categories
          }

    if SIZE_DICT:
        db = rm_nonexist_ids(db)

    t_final = time.time() - t
    print('db creation time: {} s'.format(t_final))

    t = time.time()
    with open(dst_file_path, 'w') as f:
        json.dump(db, f)
    t_final = time.time() - t
    print('JSON saving time: {} s'.format(t_final))

    #Sanity check
    #['sedan', 'truck', 'dedicated agricultural vehicle', 'jeep', 'crane truck', 'prime mover',
    # 'cement mixer', 'hatchback', 'minivan', 'pickup', 'van', 'light truck', 'bus', 'tanker', 'minibus']
    viz_random_annotation('sedan')
    viz_random_annotation('truck')
    viz_random_annotation('jeep')
    viz_random_annotation('minibus')
    viz_random_annotation('tanker')
    viz_random_annotation('cement mixer')
