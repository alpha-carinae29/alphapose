from models.fastpose import FastPose
def build_sppe_model(cfg, preset_cfg):
    args = cfg.copy()
    default_args = {
        'PRESET': preset_cfg,
    }
    for name, value in default_args.items():
        args.setdefault(name, value)
    return FastPose(**args)


def build_detection_model():
    pass
