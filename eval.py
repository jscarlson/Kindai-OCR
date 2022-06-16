# -*- coding: cp932 -*-
import torch
import torch.backends.cudnn as cudnn
from utils import dataIterator, load_dict, gen_sample, load_mapping
from encoder_decoder import Encoder_Decoder
import cv2
import numpy as np
from collections import OrderedDict
from glob import glob
import os
from tqdm import tqdm
from nltk.metrics.distance import edit_distance
import pandas as pd


def textline_evaluation(
        pairs,
        print_incorrect=True, 
        no_spaces_in_eval=True, 
        norm_edit_distance=False, 
        uncased=False
    ):

    n_correct = 0
    edit_count = 0
    length_of_data = len(pairs)
    n_chars = sum(len(gt) for gt, _ in pairs)

    for gt, pred in pairs:

        # eval w/o spaces
        gt = gt.strip() if not no_spaces_in_eval else gt.strip().replace(" ", "")
        pred = pred.strip() if not no_spaces_in_eval else pred.strip().replace(" ", "")
        if uncased:
            pred, gt = pred.lower(), gt.lower()
        
        # textline accuracy
        if pred == gt:
            n_correct += 1
        else:
            if print_incorrect:
                print(f"GT: {gt}\nPR: {pred}\n")

        # ICDAR2019 Normalized Edit Distance
        if norm_edit_distance:
            if len(gt) > len(pred):
                edit_count += edit_distance(pred, gt) / len(gt)
            else:
                edit_count += edit_distance(pred, gt) / len(pred)
        else:
            edit_count += edit_distance(pred, gt)

    accuracy = n_correct / float(length_of_data) * 100
    if norm_edit_distance:
        cer = edit_count / float(length_of_data)
    else:
        cer = edit_count / n_chars

    return accuracy, cer


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def Kindai_OCR(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = image
    h, w = image.shape[0], image.shape[1]

    if w < h:
        rate = 20.0/w
        w = int(round(w*rate))
        h = int(round(h* rate / 20.0) * 20)
    else:
        rate = 20.0/h
        w = int(round(w*rate / 20.0) * 20)
        h = int(round(h* rate))
    #print (w, h, rate)
    input_img = cv2.resize(input_img, (w,h))
    
    mat = np.zeros([1, h, w], dtype='uint8')  
    mat[0,:,:] = 0.299* input_img[:, :, 0] + 0.587 * input_img[:, :, 1] + 0.114 * input_img[:, :, 2]

    xx_pad = mat.astype(np.float32) / 255.
    xx_pad = torch.from_numpy(xx_pad[None, :, :, :])  # (1,1,H,W)
    xx_pad = xx_pad.cuda()

    with torch.no_grad():
        sample, score, _ = gen_sample(OCR, xx_pad, params, True, k=10, maxlen=600)

    score = score / np.array([len(s) for s in sample])
    ss = sample[score.argmin()]
    result = ''

    i = 0
    for vv in ss:
        if vv == 0: # <eol>
            break
        result += chr(int(worddicts_r[vv],16))
        i+=1 
    
    return result
            

if __name__ == '__main__':

    OCR_MODELPARA = "./pretrain/WAP_params.pkl"
    DICTIONARY_TARGET = "./pretrain/kindai_voc.txt"
    TEST_IMAGES = "/srv/ocr/github_repos/EasyOCR/trainer/easyocr_data/ww_test"

    label_df = pd.read_csv(os.path.join(TEST_IMAGES, "labels.csv"))
    label_dict = {fn:lab for fn, lab in zip(label_df["filename"].tolist(), label_df["words"].tolist())}

    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 5748
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 3
    params['cuda'] = True

    # load model
    OCR = Encoder_Decoder(params)
    OCR.load_state_dict(copyStateDict(torch.load(OCR_MODELPARA)))
    OCR = torch.nn.DataParallel(OCR).cuda()
    cudnn.benchmark = False
    OCR.eval()

    # load dictionary
    worddicts = load_dict(DICTIONARY_TARGET)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    # run OCR
    pairs = []
    for image_path in tqdm(glob(os.path.join(TEST_IMAGES, "*.png"))):
        result = Kindai_OCR(image_path)
        pairs.append((label_dict[os.path.basename(image_path)], result))

    accuracy, cer = textline_evaluation(pairs)

    print(accuracy, cer)