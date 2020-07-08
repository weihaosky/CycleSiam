# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join
from os import listdir
import json
import glob
import xml.etree.ElementTree as ET
import cv2, IPython, random, selectivesearch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import sys, os, io

from contextlib import contextmanager
import ctypes
import tempfile
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
            

def detect(video):
    v = dict()
    v['base_path'] = join(sub_set, video)
    v['frame'] = []
    video_base_path = join(sub_set_base_path, video)
    xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
    for xml in xmls:
        f = dict()
        xmltree = ET.parse(xml)
        size = xmltree.findall('size')[0]
        frame_sz = [int(it.text) for it in size]
        objects = xmltree.findall('object')
        objs = []

        # ========================= saliency detection ===============================
        # img = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        # # initialize OpenCV's objectness saliency detector and set the path
        # # to the input model files
        # saliency = cv2.saliency.ObjectnessBING_create()
        # modelpath = '/home/weihao/disk1/documents/github/opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel'
        # saliency.setTrainingPath(modelpath)

        # # compute the bounding box predictions used to indicate saliency
        # redirect = io.BytesIO()
        # with stdout_redirector(redirect):
        #     (success, saliencyMap) = saliency.computeSaliency(img)

        # ###### visualization
        # output = img.copy()
        # cv2.rectangle(output, (startX, startY), (endX, endY), (100, 0, 0), 2)
        # cv2.imwrite('saliency/{0}_{1}.jpg'.format(xml.split('/')[-2], xml.split('/')[-1].split('.')[0]), output)
        # ========================= saliency detection ===============================


        # ============================= selective search ==================================
        # img = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        # img_lbl, regions = selectivesearch.selective_search(img, scale=1000, sigma=0.9, min_size=10)
        # #计算一共分割了多少个原始候选区域
        # temp = set()
        # for i in range(img_lbl.shape[0]):
        #     for j in range(img_lbl.shape[1]):    
        #         temp.add(img_lbl[i,j,3]) 
        # print(len(temp))       
        # #计算利用Selective Search算法得到了多少个候选区域
        # print(len(regions))    
        # #创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
        # candidates = set()
        # for r in regions:
        #     #排除重复的候选区
        #     if r['rect'] in candidates:
        #         continue
        #     #排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)  
        #     if r['size'] < 5000:
        #         continue
        #     #排除扭曲的候选区域边框  即只保留近似正方形的
        #     # x, y, w, h = r['rect']
        #     if w / h > 2 or h / w > 2:
        #         continue
        #     candidates.add(r['rect'])

        # for x, y, w, h in candidates:
        #     print(x, y, w, h)
        #     cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
        # cv2.imwrite('img_selectivesearch.jpg', img)
        # ============================= selective search ==================================

        for i, object_iter in enumerate(objects):
            trackid = int(object_iter.find('trackid').text)
            name = (object_iter.find('name')).text
            bndbox = object_iter.find('bndbox')
            occluded = int(object_iter.find('occluded').text)
            o = dict()
            o['c'] = name
            o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                         int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

            # ================== random bbox ========================
            imw, imh = frame_sz
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            xmin_random = int(bbox[0] + random.gauss(0, 0.25) * w)
            ymin_random = int(bbox[1] + random.gauss(0, 0.25) * h)
            h_random = int(h + random.gauss(0, 0.25)*h)
            w_random = int(w + random.gauss(0, 0.25)*w)
            while xmin_random + w_random >= imw or ymin_random + h_random >= imh or \
                    xmin_random <= 0 or ymin_random <= 0 or w_random <= 1 or h_random <= 1 or \
                    w_random/h_random > 2*w/h or w_random/h_random < 0.5*w/h:   
                xmin_random = int(bbox[0] + random.gauss(0, 0.25) * w)
                ymin_random = int(bbox[1] + random.gauss(0, 0.25) * h)
                h_random = int(h + random.gauss(0, 0.25)*h)
                w_random = int(w + random.gauss(0, 0.25)*w)
            bbox_random = [xmin_random, ymin_random, xmin_random + w_random, ymin_random + h_random]
            o['bbox'] = bbox_random

            bndbox.find('xmin').text=str(bbox_random[0])
            bndbox.find('ymin').text=str(bbox_random[1])
            bndbox.find('xmax').text=str(bbox_random[2])
            bndbox.find('ymax').text=str(bbox_random[3])
            xmltree.write(xml)
            # ================== random bbox ========================

            # ========================= saliency detection ===============================
            # (startX, startY, endX, endY) = saliencyMap[i].flatten()
            # o['bbox'] = [int(startX), int(startY), int(endX), int(endY)]
            # bndbox.find('xmin').text=str(startX)
            # bndbox.find('ymin').text=str(startY)
            # bndbox.find('xmax').text=str(endX)
            # bndbox.find('ymax').text=str(endY)
            # xmltree.write(xml)
                        
            o['trackid'] = trackid
            o['occ'] = occluded
            objs.append(o)
        f['frame_sz'] = frame_sz
        f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
        f['objs'] = objs
        v['frame'].append(f)

    return v


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process vid')
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of multiple thread (default: 16)')

    args = parser.parse_args()

    VID_base_path = './ILSVRC2015'
    ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
    img_base_path = join(VID_base_path, 'Data/VID/train/')
    sub_sets = sorted({'ILSVRC2015_VID_train_0000', 'ILSVRC2015_VID_train_0001', 'ILSVRC2015_VID_train_0002', 'ILSVRC2015_VID_train_0003', 'val'})
    vid = []
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []

        # for vi, video in enumerate(videos):
        #     ret = detect(video)
        #     s.append(ret)

        with Pool(processes=args.workers) as pool:
            for ret in tqdm(pool.imap(detect, videos), desc='detect', total=len(videos), ncols=100):
                s.append(ret)
        vid.append(s)
    print('save json (raw vid info), please wait 1 min~')
    json.dump(vid, open('vid.json', 'w'), indent=4, sort_keys=True)
    print('done!')
    IPython.embed()
