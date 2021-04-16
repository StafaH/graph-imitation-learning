import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence


def get_big_image(paths, num_frames=6, out_name="seq.jpeg"):
    """ """
    H, W, C = 360, 640, 3
    num = len(paths)
    out = np.zeros((H * num + num - 1, W * num_frames, 3), dtype=np.uint8)

    for frame_i, path in enumerate(paths):
        img = Image.open(path)
        # (L,W,H,3)
        # frames = np.array([
        #     np.array(frame.copy().convert('RGB').getdata(),
        #              dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3)
        #     for frame in ImageSequence.Iterator(img)
        # ])
        frames = [
            np.array(frame.copy().convert('RGB').getdata(),
                     dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3)
            for i, frame in enumerate(ImageSequence.Iterator(img))
        ]
        frames = frames[::-len(frames) // num_frames][::-1]
        if len(frames) > num_frames:
            frames = frames[-num_frames:]
        elif len(frames) < num_frames:
            raise Exception("Bad number of frames.")

        for j, frame in enumerate(frames):
            out[frame_i * H + frame_i:(frame_i + 1) * H + frame_i,
                j * W:(j + 1) * W] = frame.copy()

    # save
    im = Image.fromarray(out)
    im.save(out_name)
    # scipy.misc.imsave('outfile.jpg', image_array)


if __name__ == "__main__":
    # paths = [
    #     "logs/rt_mlp_fix_64x3_lr/seed3_Apr15_16-39-38/run_1.gif",
    #     "logs/rt_mlp_fix_64x3_lr/seed1_Apr15_15-23-38/run_7.gif",
    #     "logs/rt_mlp_fix_64x3_lr/seed1_Apr15_15-23-38/run_0.gif",
    # ]
    paths = [
        "logs/pl_mlp_fix_64x3_lr/seed6_Apr14_22-15-15/run_6.gif",
        "logs/pl_mlp_fix_64x3_lr/seed6_Apr14_22-15-15/run_7.gif",
        "logs/pl_mlp_fix_64x3_lr/seed6_Apr14_22-15-15/run_8.gif",
    ]
    get_big_image(paths)
