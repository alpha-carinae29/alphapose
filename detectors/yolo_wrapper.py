from detectors.yolo.darknet import Darknet

import numpy as np
import cv2
import torch


class YoloWrapper:
    def __init__(self, cfg, opt):
        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_cfg = cfg.get('CONFIG', 'detector/yolo/cfg/yolov3-spp.cfg')
        self.model_weights = cfg.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights')
        self.inp_dim = cfg.get('INP_DIM', 608)
        self.nms_thres = cfg.get('NMS_THRES', 0.6)
        self.confidence = cfg.get('CONFIDENCE', 0.05)
        self.num_classes = cfg.get('NUM_CLASSES', 80)
        self.model = None

    def load_model(self):
        args = self.detector_opt

        print('Loading YOLO model..')
        self.model = Darknet(self.model_cfg)
        self.model.load_weights(self.model_weights)
        self.model.net_info['height'] = self.inp_dim
        self.model.to(args.device)
        self.model.eval()

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:-1]
        w, h = self.inp_dim, self.inp_dim
        new_w = int(orig_w * min(w / orig_w, h / orig_h))
        new_h = int(orig_h * min(w / orig_w, h / orig_h))
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        padded_image = np.full((h, w, 3), 128)

        padded_image[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w,
        :] = resized_image
        padded_image = padded_image[:, :, ::-1].transpose((2, 0, 1)).copy()
        padded_image = torch.from_numpy(padded_image).float().div(255.0).unsqueeze(0)
        return padded_image
    
    def inference(self, image):
        preprocessed_image = self.preprocess(image)
