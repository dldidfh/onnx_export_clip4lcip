import cv2 
from torchvision.transforms import (
    Compose,
    Normalize,
    ToPILImage,
    ToTensor,
    Resize,
    CenterCrop
)
from PIL import Image 

import numpy as np 
origin_transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

normalized_transform = Compose(
            [   ToPILImage(),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),),
                ])
non_normalized_transform = Compose(
            [   ToPILImage(),
                ToTensor(),
                ])

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        self.origin_shape = shape
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        self.ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        self.dw, self.dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, self.stride), np.mod(
                self.dh, self.stride
            )  # wh padding
        elif self.scaleFill:  # stretch
            self.dw, self.dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            self.ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (
                labels["ratio_pad"],
                (self.dw, self.dh),
            )  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, self.ratio, self.dw, self.dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

    def get_origin_size_bbox(self, resized_bboxes: np.ndarray):
        resized_bboxes[:, [0, 2]] = (resized_bboxes[:, [0, 2]] - self.dw) * (
            1 / self.ratio[0]
        )
        resized_bboxes[:, [1, 3]] = (resized_bboxes[:, [1, 3]] - self.dh) * (
            1 / self.ratio[1]
        )
        return resized_bboxes
