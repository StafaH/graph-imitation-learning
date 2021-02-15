#!/usr/bin/env python

"""visualize_bbox_coco.py: Visualize the annotations from a COCO style dataset

    Note: Requires that that COCO annotation have the exact same filenames as the files in the data folder
    Note: pycoco implementation of visualizing annotations as found at https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

    Arguments:
        Required:
        -f, --file = location of annotation file
        -d, --data = location of data

        Optional:
        -i, --index = index of image to visualize if more than one (default = 0)

    Usage:  visualize_bbox_coco.py [-h] -f FILE -d DATA [-i INDEX]
    Example usage: python src/visualize/visualize_bbox_coco.py -f data/processed/incision_1/annotations/instances_default.json -d data/processed/incision_1/ 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import isfile, isdir

from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description='Seperate video file into individual frames')
    parser.add_argument('-f', '--file', type=str, required=True, help='name and location of file you want to parse')
    parser.add_argument('-d', '--data', type=str, required=True, help='location of data')
    parser.add_argument('-i', '--index', type=int, default=0, help='index of image you want to see (default = 0)')
    args = parser.parse_args()

    annotation_file = args.file
    data_folder = args.data
    image_index = args.index 

    # Check if input file exists
    if not isfile(annotation_file):
        print('Annotation file does not exist')
        return
    
    if not isdir(data_folder):
        print('Data folder does not exist')
        return


    # initialize COCO api for instance annotations
    coco = COCO(annotation_file)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    #nms = set([cat['supercategory'] for cat in cats])
    #print('COCO supercategories: \n{}'.format(' '.join(nms)))
    
    # Get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms = ['Tool'])
    imgIds = coco.getImgIds(catIds = catIds )

    img = coco.loadImgs(imgIds[image_index])[0]

    # Use OpenCV to read the image file and draw to matplotlib
    print('Reading ' + img['file_name'])
    read_image = cv2.imread(data_folder + img['file_name'])
    plt.axis('off')
    plt.imshow(cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB))

    # Get annotaton from COCO and draw it matplotlib
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=True)
    plt.show()

if __name__ == '__main__':
    main()