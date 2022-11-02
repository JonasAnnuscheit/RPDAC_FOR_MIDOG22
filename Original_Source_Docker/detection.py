import logging
import torch
from queue import Queue, Empty
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from util.nms_WSI import nms_patch, nms
from util.object_detection_helper import create_anchors, process_output, rescale_box
#from fastai.vision.learner import create_body
#from fastai.vision import models
#import model
#from model import RetinaNetDA

#import sys
#import os
#currentpath = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(currentpath)
#sys.path.append(currentpath+'/models')


import torch
import models
import models.yolo
import models.common

from models.yolo import Model
from models.common import AutoShape
from skimage.color import rgb2gray, rgb2hed
import pandas as pd

def load_automodels(paths):
  automodels = []
  for path_to_model in paths:
      # Load structure of the model
      device = 'cuda'
      cfg = 'models/MYyolov5s0.1concatT.yaml'
      model = Model(cfg, ch=3, nc=1, anchors=3).to(device)  # create
      
      # load weights
      print(path_to_model)
      csd = torch.load(path_to_model, map_location='cpu')['model'].float().state_dict()
      model.load_state_dict(csd, strict=False)  # load
      model = model.eval()
      automodel = AutoShape(model)
      automodels.append(automodel)
  return automodels

def to_gray(img):
    img = rgb2gray(img)
    if np.all(img<=1):
        img = np.array(img*255.9,dtype=np.uint8)
    else:
        img = np.array(img,dtype=np.uint8)
    img = np.stack([img,img,img], axis=-1)
    return img
def to_hed(img):
    hed = rgb2hed(img)
    img = np.array(np.clip(hed * 1280,0,255), dtype=np.uint8)
    return img
def to_he(img):
    hed = rgb2hed(img)
    hed[...,2] = 0
    img = np.array(np.clip(hed * 1280,0,255), dtype=np.uint8)
    return img

class MyMitosisDetection:
    def __init__(self, path_model, size, batchsize, detect_threshold = 0.64, nms_threshold = 0.4):
        self.size=1280
        self.batchsize=1
        self.detect_threshold = 0.4
        self.nms_threshold = 0.4
        """
        # network parameters
        self.detect_thresh = detect_threshold
        self.nms_thresh = nms_threshold
        encoder = create_body(models.resnet18, False, -2)
        scales = [0.2, 0.4, 0.6, 0.8, 1.0]
        ratios = [1]
        sizes = [(64, 64), (32, 32), (16, 16)]
        self.model = RetinaNetDA.RetinaNetDA(encoder, n_classes=2, n_domains=4,  n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)
        self.path_model = path_model
        self.size = size
        self.batchsize = batchsize
        self.mean = None
        self.std = None
        self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        """
        pass

 

    def load_model(self):
        path_to_models = [
            'yolo_models/lrl.pt',
            #'yolo_models/da2.pt',
            #'yolo_models/da3.pt',
            #'yolo_models/he.pt',
            #'yolo_models/hed.pt',
        ]
        self.automodels = load_automodels(path_to_models)

        self.models_input_manipulatorfunctions = [
            None,
            #None,
            #None,
            #to_gray,
            #to_he,
            #to_hed,
        ]
        self.model_pred_divider = [0.402] #,0.394,0.368]
        return True

    def create_predictions(self, img):
        automodels = self.automodels
        models_input_manipulatorfunctions = self.models_input_manipulatorfunctions
        img = img[...,:3]
        if np.any(img>1):
            img = np.array(img,dtype=np.uint8)
        else:
            img = np.array(img*255.99,dtype=np.uint8)
        ps = []
        for i,automodel in enumerate(automodels):
            _img = np.copy(img)
            if models_input_manipulatorfunctions[i]:
                _img = models_input_manipulatorfunctions[i](_img)
            detection = automodel(_img, 1280)
            pred = detection.pred[0].cpu().numpy()
            x = (pred[:,0] + pred[:,2]) / 2
            y = (pred[:,1] + pred[:,3]) / 2
            prob = pred[:,4] / (self.model_pred_divider[i]*2)
            p = np.array([[i]*len(x),x,y,prob]).T
            ps.extend(p)
        for i,automodel in enumerate(automodels):
            _img = np.copy(img)[::-1,:,:]
            if models_input_manipulatorfunctions[i]:
                _img = models_input_manipulatorfunctions[i](_img)
            detection = automodel(_img, 1280)
            pred = detection.pred[0].cpu().numpy()
            x = (pred[:,0] + pred[:,2]) / 2
            y = 1280- ((pred[:,1] + pred[:,3]) / 2)
            prob = pred[:,4] / (self.model_pred_divider[i]*2)
            p = np.array([[i]*len(x),x,y,prob]).T
            ps.extend(p)
            
        for i,automodel in enumerate(automodels):
            _img = np.copy(img)[::,::-1,:]
            if models_input_manipulatorfunctions[i]:
                _img = models_input_manipulatorfunctions[i](_img)
            detection = automodel(_img, 1280)
            pred = detection.pred[0].cpu().numpy()
            x = 1280- ((pred[:,0] + pred[:,2]) / 2)
            y =  ((pred[:,1] + pred[:,3]) / 2)
            prob = pred[:,4] / (self.model_pred_divider[i]*2)
            p = np.array([[i]*len(x),x,y,prob]).T
            ps.extend(p)
            
        for i,automodel in enumerate(automodels):
            _img = np.copy(img)[::-1,::-1,:]
            if models_input_manipulatorfunctions[i]:
                _img = models_input_manipulatorfunctions[i](_img)
            detection = automodel(_img, 1280)
            pred = detection.pred[0].cpu().numpy()
            x = 1280-((pred[:,0] + pred[:,2]) / 2)
            y = 1280-((pred[:,1] + pred[:,3]) / 2)
            prob = pred[:,4] / (self.model_pred_divider[i]*2)
            p = np.array([[i]*len(x),x,y,prob]).T
            ps.extend(p)
        ps = np.array(ps)
        
        return ps

    def create_prediction_table(self, ps):
        print('Start merging predictions (in create_prediction_table)')
        automodels = self.automodels
        ps = np.array(ps)
        print('ps:', ps)
        points = []
        if len(ps) > 0:
            print(len(ps))
            dis = ((ps[:,1:2]-ps[:,1:2].T)**2+(ps[:,2:3]-ps[:,2:3].T)**2)**0.5 < 20
            print('dis', dis)
            for i, d in enumerate(dis):
                foundself = False
                p = []
                for j, v in enumerate(d):
                    if i==j:
                        foundself = True
                    if not foundself and v: # already in the list
                        break
                    elif v:
                        p.append(j)
                if p:
                    points.append(p)
        print('points:', points)
        pred_table = np.zeros([len(points),2+2*len(automodels)])

        for i, p in enumerate(points):
            pred_table[i,0] = np.mean(ps[p][:,1]) # X
            pred_table[i,1] = np.mean(ps[p][:,2]) # Y
            for j, _p in enumerate(p):
                pred_table[i,int(2+2*ps[p][j,0])] = ps[p][j,3]
                pred_table[i,int(2+2*ps[p][j,0])+1] = ((pred_table[i,0]-ps[p][j,1])**2+(pred_table[i,1]-ps[p][j,2])**2)**0.5
        print('pred_table:', pred_table)
        print()
        return pred_table



    def process_image(self, input_image):
        n_patches = 0
        img_dimensions = np.array(input_image.shape)
        print('start: process_image, the img dim is:', img_dimensions)

        img_is_to_small = False

        if img_dimensions[0] < 1280:
            img_dimensions[0] = 1280
            img_is_to_small = True
        if img_dimensions[1] < 1280:
            img_dimensions[1] = 1280
            img_is_to_small = True
        if img_is_to_small:
            print('Warning IMG is to small, adding boarder to the image:', input_image.shape)
            tmp_img = np.zeros([img_dimensions[0],img_dimensions[1],3], dtype=input_image.dtype)
            tmp_img[:input_image.shape[0],:input_image.shape[1]] = input_image[:,:,:3]
            input_image = tmp_img

        all_predictions = []

        # create overlapping patches for the whole image
        for x in np.arange(0, img_dimensions[1], int(0.9 * self.size)):
            for y in np.arange(0, img_dimensions[0], int(0.9 * self.size)):
                # last patch shall reach just up to the last pixel
                if (x+self.size>img_dimensions[1]):
                    x = img_dimensions[1]-1280

                if (y+self.size>img_dimensions[0]):
                    y = img_dimensions[0]-1280

                img = input_image[y:y+1280,x:x+1280]

                #x,y,img
                pred = self.create_predictions(img)
                if len(pred) > 0:
                    pred[:,1] += x
                    pred[:,2] += y
                    all_predictions.extend(pred)
        print('Finish with prediction all tiles, num of overlapping preds:', len(all_predictions))
        
        pred_table = self.create_prediction_table(all_predictions)
        pred_table[:,2] = np.mean(pred_table[:,2::2],axis=1)

        pred_table = pred_table[:,:3] # use only first model

        # TODO PREPROCESSING
        all_predictions = np.array(pred_table)
        all_predictions = all_predictions.reshape([-1,3])
        print('all_predictions',all_predictions)

        return all_predictions

    def get_batch(self, queue_patches):
        batch_images = np.zeros((self.batchsize, 3, self.size, self.size))
        batch_x = np.zeros(self.batchsize, dtype=int)
        batch_y = np.zeros(self.batchsize, dtype=int)
        for i_batch in range(self.batchsize):
            if queue_patches.qsize() > 0:
                status, batch_x[i_batch], batch_y[i_batch], image = queue_patches.get()
                x_start, y_start = int(batch_x[i_batch]), int(batch_y[i_batch])

                cur_patch = image[y_start:y_start+self.size, x_start:x_start+self.size] / 255.
                batch_images[i_batch] = cur_patch.transpose(2, 0, 1)[0:3]
            else:
                batch_images = batch_images[:i_batch]
                batch_x = batch_x[:i_batch]
                batch_y = batch_y[:i_batch]
                break
        torch_batch = torch.from_numpy(batch_images.astype(np.float32, copy=False)).to(self.device)
        for p in range(torch_batch.shape[0]):
            torch_batch[p] = transforms.Normalize(self.mean, self.std)(torch_batch[p])
        return torch_batch, batch_x, batch_y

    def postprocess_patch(self, cur_bbox_pred, cur_class_pred, x_real, y_real):
        cur_patch_boxes = []

        for clas_pred, bbox_pred in zip(cur_class_pred[None, :, :], cur_bbox_pred[None, :, :], ):
            modelOutput = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            bbox_pred, scores, preds = [modelOutput[x] for x in ['bbox_pred', 'scores', 'preds']]

            if bbox_pred is not None:
                # Perform nms per patch to reduce computation effort for the whole image (optional)
                to_keep = nms_patch(bbox_pred, scores, self.nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[
                    to_keep].cpu()

                t_sz = torch.Tensor([[self.size, self.size]]).float()

                bbox_pred = rescale_box(bbox_pred, t_sz)

                for box, pred, score in zip(bbox_pred, preds, scores):
                    y_box, x_box = box[:2]
                    h, w = box[2:4]

                    cur_patch_boxes.append(
                        np.array([x_box + x_real, y_box + y_real,
                                  x_box + x_real + w, y_box + y_real + h,
                                  pred, score]))

        return cur_patch_boxes


