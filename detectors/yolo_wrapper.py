from detectors.yolo.darknet import Darknet
from detectors.yolo.util import unique, bbox_iou
import numpy as np
import cv2
import torch
import platform

if platform.system() != 'Windows':
    from detectors.nms import nms_wrapper


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
        return padded_image, torch.FloatTensor((orig_w, orig_h)).repeat(1, 2)

    def inference(self, image):
        preprocessed_image, orig_dim = self.preprocess(image)

        if not self.model:
            self.load_model()
        with torch.no_grad():
            preprocessed_image = preprocessed_image.to(self.detector_opt.device)
            prediction = self.model(preprocessed_image, args=self.detector_opt)
            detections = self.postprocess(prediction, self.confidence, self.num_classes, orig_dim=orig_dim, nms=True, nms_conf=self.nms_thres)
            if isinstance(detections, int) or detections.shape[0] == 0:
                return 0
            return detections

    def postprocess(self, prediction, confidence, num_classes, orig_dim, nms=True, nms_conf=0.4):
        conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
        prediction = prediction * conf_mask

        try:
            ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
        except:
            return 0

            # the 3rd channel of prediction: (xc,yc,w,h)->(x1,y1,x2,y2)
        box_a = prediction.new(prediction.shape)
        box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_a[:, :, :4]

        batch_size = prediction.size(0)

        output = prediction.new(1, prediction.size(2) + 1)
        write = False
        num = 0
        for ind in range(batch_size):
            # select the image from the batch
            image_pred = prediction[ind]

            # Get the class having maximum score, and the index of that class
            # Get rid of num_classes softmax scores
            # Add the class index and the class score of class having maximum score
            max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            # image_pred:(n,(x1,y1,x2,y2,c,s,idx of cls))
            image_pred = torch.cat(seq, 1)

            # Get rid of the zero entries
            non_zero_ind = (torch.nonzero(image_pred[:, 4]))

            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

            # Get the various classes detected in the image
            try:
                img_classes = unique(image_pred_[:, -1])
            except:
                continue

            # WE will do NMS classwise
            # print(img_classes)
            for cls in img_classes:
                if cls != 0:
                    continue
                # get the detections with one particular class
                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                # sort the detections such that the entry with the maximum objectness
                # confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)

                # if nms has to be done
                if nms:
                    if platform.system() != 'Windows':
                        # We use faster rcnn implementation of nms (soft nms is optional)
                        nms_op = getattr(nms_wrapper, 'nms')
                        # nms_op input:(n,(x1,y1,x2,y2,c))
                        # nms_op output: input[inds,:], inds
                        _, inds = nms_op(image_pred_class[:, :5], nms_conf)

                        image_pred_class = image_pred_class[inds]
                    else:
                        # Perform non-maximum suppression
                        max_detections = []
                        while image_pred_class.size(0):
                            # Get detection with highest confidence and save as max detection
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            # Stop if we're at the last detection
                            if len(image_pred_class) == 1:
                                break
                            # Get the IOUs for all boxes with lower confidence
                            ious = bbox_iou(max_detections[-1], image_pred_class[1:], self.detector_opt)
                            # Remove detections with IoU >= NMS threshold
                            image_pred_class = image_pred_class[1:][ious < nms_conf]

                        image_pred_class = torch.cat(max_detections).data

                # Concatenate the batch_id of the image to the detection
                # this helps us identify which image does the detection correspond to
                # We use a linear straucture to hold ALL the detections from the batch
                # the batch_dim is flattened
                # batch is identified by extra batch column

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))
                num += 1
        if not num:
            return 0
        output = output.cpu()
        orig_dim_list = torch.index_select(orig_dim, 0, output[:, 0].long())
        scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, orig_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, orig_dim_list[i, 1])
        # output:(n,(batch_ind,x1,y1,x2,y2,c,s,idx of cls))
        return output
