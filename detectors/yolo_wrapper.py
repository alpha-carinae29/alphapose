from detectors.yolo.darknet import Darknet


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
