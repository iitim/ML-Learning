import os

import cv2
import numpy as np
import imageio
from PIL import Image


def video_to_images(src_vdo, dst_dir, limited_frame):
    vc = cv2.VideoCapture(src_vdo)
    count = 1
    success = 1
    video_name = os.path.split(src_vdo)[1].split(".")[0]

    if limited_frame > 1:
        dst_dir = os.path.join(dst_dir, video_name)
        os.makedirs(dst_dir, exist_ok=True)

    while success and count <= limited_frame:
        success, frame = vc.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil = im_pil.resize((224, 224), Image.NEAREST)
        im_pil = np.array(im_pil)

        filename = f"{str(count)}.jpg" if limited_frame > 1 else f"{video_name}.jpg"
        imageio.imwrite(os.path.join(dst_dir, filename), frame)
        count += 1
    vc.release()


def batch_process_videos(src_dir, dst_dir):
    data_groups = ('train', 'validation', 'test')
    data_classes = ('real', 'synthesis')
    os.makedirs(dst_dir, exist_ok=True)

    for data_group in data_groups:
        src_group_path = os.path.join(src_dir, data_group)
        dst_group_path = os.path.join(dst_dir, data_group)
        os.makedirs(dst_group_path, exist_ok=True)

        for data_class in data_classes:
            src_class_path = os.path.join(src_group_path, data_class)
            dst_class_path = os.path.join(dst_group_path, data_class)
            os.makedirs(dst_class_path, exist_ok=True)

            for file in os.listdir(src_class_path):
                filename = os.path.split(file)[1]
                if filename.endswith(".mp4"):
                    src_vdo_file = os.path.join(src_class_path, filename)
                    video_to_images(src_vdo_file, dst_class_path, 1)


batch_process_videos('../formatted_dataset', '../formatted_img_dataset')
