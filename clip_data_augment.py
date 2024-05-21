import os
from dataclasses import dataclass

import clip
import numpy as np
import torch
from PIL import Image

from model import H14_NSFW_Detector


@dataclass
class ImageResult:
    image_name: str
    feature: np.ndarray
    nsfw_result: int


class DataChecker:
    def __init__(self, model_weight_path, dim=768, ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.nsfw_model = H14_NSFW_Detector(dim, 5)
        self.nsfw_model.load_state_dict(torch.load(model_weight_path))
        self.nsfw_model = self.nsfw_model.cuda()
        self.nsfw_model.eval()

    @staticmethod
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def predict_result(self, image_save_dir: str):
        file_list = os.listdir(image_save_dir)
        batch_size = 16
        error_load_file_list = []
        result = []
        for i in range(0, len(file_list), batch_size):
            batch_image_list = []
            batch_file_list = []
            for file in file_list[i:i + batch_size]:
                try:
                    image = Image.open(os.path.join(image_save_dir, file))
                    image = self.clip_preprocess(image)
                    batch_image_list.append(image)
                    batch_file_list.append(file)
                except Exception as e:
                    print(f"error load file:{file}")
                    error_load_file_list.append(file)
            image_tensor = torch.cat(batch_image_list, dim=0)
            with torch.no_grad():
                image_features = self.clip_model(image_tensor.cuda())
                norm_features = self.normalized(image_features.cpu().numpy())
                nsfw_result = self.nsfw_model(torch.tensor(norm_features.astype(np.float32)).cuda())
                soft_result = torch.argmax(nsfw_result, dim=1).cpu().numpy()
            for file_name, nsfw, feature in zip(batch_file_list, soft_result, norm_features):
                result.append(ImageResult(file_name, feature, nsfw))
        return result, error_load_file_list
