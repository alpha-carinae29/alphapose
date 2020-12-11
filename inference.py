import argparse
import torch
import os
import cv2

from utils import config_parser
from utils.bbox import box_to_center_scale, center_scale_to_box
from utils.transformations import get_affine_transform, im_to_torch
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
        self.detection_model = builder.build_detection_model(self.args)
        self.detection_model.load_model()
        self._aspect_ratio = float(self._input_size[1]) / self._input_size[0]

    def inference(self, image):
        detections = self.detection_model.inference(image)
        inps, cropped_boxes = self.transform_detections(image, detections)

    def transform_detections(self, image, dets):
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transform_single_detection(image, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)
        return inps, cropped_boxes

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

    def visualize(self, image, poses):
        pass


def inference():
    output_path = args.output_path
    model = AlphaPose(args, cfg)
    input_path = args.input_path
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    poses = model.inference(image)
    output_image = model.visualize(image, poses)
    cv2.imwrite(os.path.join(output_path, "alphapose_" + os.path.basename(input_path)), output_image)


if __name__ == "__main__":
    inference()
