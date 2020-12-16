import argparse
import torch
import os
import cv2
import numpy as np
import time

from utils import config_parser
from utils.bbox import box_to_center_scale, center_scale_to_box
from utils.transformations import get_affine_transform, transform_preds, im_to_torch, get_max_pred
from utils.pose_nms import pose_nms
from builders import builder

parser = argparse.ArgumentParser(description="Simplified Demo of AlphaPose for Single Image")
parser.add_argument("--input_path", type=str, required=True, help="path of the input image")
parser.add_argument("--output_path", type=str, default="outputs/", help="path for the output result image")
parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="config file")
parser.add_argument("--checkpoint", type=str, default="pretrained_models/fast_res50_256x192.pth",
                    help="checkpoint file of pose estimator")
parser.add_argument("--gpus", type=str, default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument("--device", )
args = parser.parse_args()
cfg = config_parser.parse(args.config_file)
args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")


class AlphaPose:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self.pose_model = builder.build_sppe_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_model.to(args.device)
        self.pose_model.eval()
        self.detection_model = builder.build_detection_model(self.args)
        self.detection_model.load_model()
        self._aspect_ratio = float(self._input_size[1]) / self._input_size[0]
        self.hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.detection_time = []
        self.transformation_time = []
        self.pose_inference_time = []
        self.post_process_time = []
        self.total_time = []
    def inference(self, image):
        start_time = time.perf_counter()
        detections = self.detection_model.inference(image)
        finish_time_det = time.perf_counter()
        with torch.no_grad():
            inps, cropped_boxes, boxes, scores, ids = self.transform_detections(image, detections)
            finish_time_trans = time.perf_counter()
            inps = inps.to(self.args.device)
            hm = self.pose_model(inps)
            finish_time_pose = time.perf_counter()
            poses = self.post_process(hm, cropped_boxes, boxes, scores, ids)
            finish_time_post = time.perf_counter()
            detection_time = round(finish_time_det - start_time, 6)
            trans_time = round(finish_time_trans - finish_time_det, 6)
            pose_time = round(finish_time_pose - finish_time_trans, 6)
            post_time = round(finish_time_post - finish_time_pose, 6)
            total_time = round(finish_time_post - start_time, 6)
            self.detection_time.append(detection_time)
            self.transformation_time.append(trans_time)
            self.pose_inference_time.append(pose_time)
            self.post_process_time.append(post_time)
            self.total_time.append(total_time)
            print("detectoin : {}, transformation: {}, pose : {}, post processing: {}".format(
                detection_time, trans_time, pose_time, post_time
            ))
            return poses

    def transform_detections(self, image, dets):
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = torch.zeros(scores.shape)
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transform_single_detection(image, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)
        return inps, cropped_boxes, boxes, scores, ids

    def transform_single_detection(self, image, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        inp_h, inp_w = self._input_size
        img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def post_process(self, hm, cropped_boxes, boxes, scores, ids):
        assert hm.dim() == 4
        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i][self.eval_joints], bbox, hm_shape=self.hm_size,
                                                           norm_type=None)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(boxes, scores, ids, preds_img, preds_scores, 0)

        _result = []
        for k in range(len(scores)):
            _result.append(
                {
                    'keypoints': preds_img[k],
                    'kp_score': preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx': ids[k],
                    'bbox': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                }
            )
        return _result

    def visualize(self, image, poses):
        pass

    def heatmap_to_coord(self, hms, bbox, hms_flip=None, **kwargs):
        if hms_flip is not None:
            hms = (hms + hms_flip) / 2
        if not isinstance(hms, np.ndarray):
            hms = hms.cpu().data.numpy()
        coords, maxvals = get_max_pred(hms)

        hm_h = hms.shape[1]
        hm_w = hms.shape[2]

        # post-processing
        for p in range(coords.shape[0]):
            hm = hms[p]
            px = int(round(float(coords[p][0])))
            py = int(round(float(coords[p][1])))
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]))
                coords[p] += np.sign(diff) * .25

        preds = np.zeros_like(coords)

        # transform bbox to scale
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        center = np.array([xmin + w * 0.5, ymin + h * 0.5])
        scale = np.array([w, h])
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center, scale,
                                       [hm_w, hm_h])

        return preds, maxvals


def inference():
    output_path = args.output_path
    model = AlphaPose(args, cfg)
    input_path = args.input_path
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _ in range(2000):
        poses = model.inference(image)
    print("=================================================================")
    detection_time = sum(model.detection_time) / len(model.detection_time)
    trans_time = sum(model.transformation_time) / len(model.transformation_time)
    pose_time = sum(model.pose_inference_time) / len(model.pose_inference_time)
    post_time = sum(model.post_process_time) / len(model.post_process_time)
    total_time = sum(model.total_time) / len(model.total_time)
    print("detectoin : {}, transformation: {}, pose : {}, post processing: {}, total time : {}".format(
        detection_time, trans_time, pose_time, post_time, total_time
    ))
    print("==================================================================")
    print("                            FPS                                   ")
    print("==================================================================")
    print("detectoin : {}, transformation: {}, pose : {}, post processing: {}, total time : {}".format(
        (1 / detection_time), 1 / trans_time, 1 / pose_time, 1 / post_time, 1 / total_time
    ))


    output_image = model.visualize(image, poses)
    cv2.imwrite(os.path.join(output_path, "alphapose_" + os.path.basename(input_path)), output_image)


if __name__ == "__main__":
    inference()
