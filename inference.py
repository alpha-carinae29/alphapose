import argparse
import torch
import os
import cv2

parser = argparse.ArgumentParser(description="Simplified Demo of AlphaPose for Single Image")
parser.add_argument("--input_path", type=str, required=True, help="path of the input image")
parser.add_argument("--output_path", type=str, default="outputs/", help="path for the output result image")
args = parser.parse_args()


class AlphaPose:
    def __init__(self, args):
        pass

    def inference(self, image):
        pass

    def visualize(self, image, poses):
        pass


def inference():
    output_path = args.output_path
    model = AlphaPose(args)
    input_path = args.input_path
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    poses = model.inference(image)
    output_image = model.visualize(image, poses)
    cv2.imwrite(os.path.join(output_path, "alphapose_" + os.path.basename(input_path)), output_image)


if __name__ == "__main__":
    inference()
