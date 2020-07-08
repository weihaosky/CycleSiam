# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, isdir
from os import mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import time
import sys

import random, os, io
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


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_like_SiamFCx(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def crop_xml(xml, sub_set_crop_path, instanc_size=511):
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')

    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    # ========================= saliency detection ===============================
    img = im
    saliency = cv2.saliency.ObjectnessBING_create()
    modelpath = '/home/weihao/disk1/documents/github/opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel'
    saliency.setTrainingPath(modelpath)
    redirect = io.BytesIO()
    with stdout_redirector(redirect):
        (success, saliencyMap) = saliency.computeSaliency(img)
    # ========================= saliency detection ===============================

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

        # z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        # x = crop_like_SiamFCx(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        # cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)

        # ================== random bbox ========================
        # imh, imw = im.shape[:2]
        # h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        # xmin_random = int(bbox[0] + random.gauss(0, 0.5) * w)
        # ymin_random = int(bbox[1] + random.gauss(0, 0.5) * h)
        # h_random = int(h + random.gauss(0, 0.5)*h)
        # w_random = int(w + random.gauss(0, 0.5)*w)
        # Count = 0
        # while xmin_random + w_random >= imw or ymin_random + h_random >= imh or \
        #         xmin_random <= 0 or ymin_random <= 0 or w_random <= 1 or h_random <= 1:   
        #     xmin_random = int(bbox[0] + random.gauss(0, 0.5) * w)
        #     ymin_random = int(bbox[1] + random.gauss(0, 0.5) * h)
        #     h_random = int(h + random.gauss(0, 0.5)*h)
        #     w_random = int(w + random.gauss(0, 0.5)*w)
        #     Count += 1
        #     if Count >= 1000: break
        # if Count < 1000:
        #     bbox_random = [xmin_random, ymin_random, xmin_random + w_random, ymin_random + h_random]
        #     bndbox.find('xmin').text=str(bbox_random[0])
        #     bndbox.find('ymin').text=str(bbox_random[1])
        #     bndbox.find('xmax').text=str(bbox_random[2])
        #     bndbox.find('ymax').text=str(bbox_random[3])
        #     xmltree.write(xml)
        # else:
        #     bbox_random = bbox
        #     print('non random bbox：', bbox, xml)
        # ================== random bbox ========================

        # ========================= saliency detection ===============================
        (startX, startY, endX, endY) = saliencyMap[id].flatten()
        bbox_salient = [int(startX), int(startY), int(endX), int(endY)]
        bndbox.find('xmin').text=str(startX)
        bndbox.find('ymin').text=str(startY)
        bndbox.find('xmax').text=str(endX)
        bndbox.find('ymax').text=str(endY)
        xmltree.write(xml)

        x = crop_like_SiamFCx(im, bbox_salient, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)


def main(instanc_size=511, num_threads=24):
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)
    VID_base_path = './ILSVRC2015'
    ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('ILSVRC2013_train', 'ILSVRC2013_train_extra0', 'ILSVRC2013_train_extra1', 'ILSVRC2013_train_extra2', 'ILSVRC2013_train_extra3', 'ILSVRC2013_train_extra4', 'ILSVRC2013_train_extra5', 'ILSVRC2013_train_extra6', 'ILSVRC2013_train_extra7', 'ILSVRC2013_train_extra8', 'ILSVRC2013_train_extra9', 'ILSVRC2013_train_extra10', 'ILSVRC2014_train_0000', 'ILSVRC2014_train_0001','ILSVRC2014_train_0002','ILSVRC2014_train_0003','ILSVRC2014_train_0004','ILSVRC2014_train_0005','ILSVRC2014_train_0006')
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        if 'ILSVRC2013_train' == sub_set:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))

        n_imgs = len(xmls)
        sub_set_crop_path = join(crop_path, sub_set)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_xml, xml, sub_set_crop_path, instanc_size) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
