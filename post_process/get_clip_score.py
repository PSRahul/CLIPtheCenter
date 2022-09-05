import os.path
import sys

import clip
import torch
from PIL import Image

image_root = "/home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA_CA/support_images/base_classes/val/"
import numpy as np

image_list = ["aeroplane.png",
              "dog.png",
              "sheep.png"]


def main():
    clip_embeddings = np.zeros((len(image_list), 512))
    for index, image_path in enumerate(image_list):
        with torch.no_grad():
            clip_model, clip_preprocess = clip.load("ViT-B/16", device="cuda")
            clip_model = clip_model.cuda().eval()
            image = Image.open(os.path.join(image_root, image_path))
            image = clip_preprocess(image).unsqueeze(0)
            image_clip_embedding = clip_model.encode_image(image.cuda())
            image_clip_embedding = image_clip_embedding.cpu().numpy()
            clip_embeddings[index] = image_clip_embedding

    np.save(os.path.join(image_root, "clip_embeddings.npy"), clip_embeddings)


if __name__ == "__main__":
    sys.exit(main())
