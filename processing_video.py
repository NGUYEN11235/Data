import pandas as pd
import os
import cv2
import pafy
from PIL import Image

from multiprocessing import Pool
import numpy as np
import torchvision.transforms.functional as functional


df = pd.read_pickle('golfDB.pkl')
yt_video_dir = '../../database/videos/'


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


# class Resize(object):
#     def __init__(self, sizes, max_size=None):
#         assert isinstance(sizes, (list, tuple))
#         self.sizes = sizes
#         self.max_size = max_size
#
#     def __call__(self, img):
#         size = self.sizes
#         return resize(img, size, self.max_size)


def preprocess_videos(index, dim=640, face_on=True):
    """
    Extracts relevant frames from youtube videos
    """
    df = pd.read_pickle('golfDB.pkl')
    yt_video_dir = '../../database/videos/'
    if face_on:
        df = df.loc[df.slow == 1]
        df = df.loc[df.view == 'face-on']
        df.index = range(len(df))
        # df = df.drop(['split'], axis=1)

    table = df.loc[df.index == index]

    bbox = table['bbox']
    events = list(table['events'])
    path = 'videos_data/'

    if not os.path.isfile(os.path.join(path, "{}.mp4".format(index))):
        print('Processing annotation id {}'.format(index))
        url = "https://www.youtube.com/watch?v=" + str(list(table['youtube_id'])[0])
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter("data/videos_160/{}.mp4".format(anno_id),
        #                       fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))
        out = cv2.VideoWriter("videos_data/{}.mp4".format(index), fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))

        x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * list(bbox)[0][0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * list(bbox)[0][1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * list(bbox)[0][2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * list(bbox)[0][3])
        count = 0
        success, image = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, events[0][0])
        while success:
            count += 1

            if count >= events[0][0] and count <= events[0][-1]:
                    crop_img = image[y:y + h, x:x + w]
                    crop_size = crop_img.shape[:2]
                    # ratio = dim / max(crop_size)
                    # new_size = tuple([int(x*ratio) for x in crop_size])
                    # resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
                    # delta_w = dim - new_size[1]
                    # delta_h = dim - new_size[0]
                    # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    # left, right = delta_w // 2, delta_w - (delta_w // 2)
                    # b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                    #                            value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)

                    b_img = cv2.resize(crop_img, (640, 640))

                    # cv2.imshow("video", b_img)
                    cv2.waitKey(1)
                    out.write(b_img)
            if count > events[0][-1]:
                break
            success, image = cap.read()
    else:
        print('Annotation id {} already completed for size {}'.format(index, dim))



if __name__ == '__main__':
    df = pd.read_pickle('golfDB.pkl')
    yt_video_dir = '../../database/videos/'
    path = 'videos_data/'
    if not os.path.exists(path):
        os.mkdir(path)
    # preprocess_videos(1, dim=640)
    for i in range(461):
        # print(i)
        preprocess_videos(i)
    # p = Pool(6)
    # p.map(preprocess_videos, df.id)