# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Based on the code by NielsRogge from HuggingFace
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Multimodal_Autoencoding.ipynb

import os
import re
import ssl
import tempfile
import subprocess
import getpass
from urllib import request

import cv2
import torch
import poptorch
import numpy as np
import scipy.io.wavfile
from transformers import PerceiverConfig

from models.multimodal_modelling import IPUPerceiverForMultimodalAutoencoding


PRECISION = 'fp16'
AUDIO_SAMPLES_PER_FRAME = 48000 // 25

PROFILE_DIR = f'/home/{getpass.getuser()}/reports/multimodal_perceiver_{PRECISION}'
EXECUTABLE_CACHE_DIR = './exec_cache'

# Utilities to fetch videos from UCF101 dataset
UCF_ROOT = 'https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/'
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()


# Video utilities
def list_ucf_videos():
    """Lists videos available in UCF101 dataset."""
    global _VIDEO_LIST
    if not _VIDEO_LIST:
        index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode('utf-8')
        videos = re.findall('(v_[\w_]+\.avi)', index)
        _VIDEO_LIST = sorted(set(videos))
    return list(_VIDEO_LIST)


def fetch_ucf_video(video):
    """Fetchs a video and cache into local filesystem."""
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print('Fetching %s => %s' % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


# Prediction
def autoencode_video(model, images, audio):
    inputs = {'image': torch.from_numpy(np.moveaxis(images, -1, 2)).float(),
              'audio': torch.from_numpy(audio),
              'label': torch.zeros((images.shape[0], 700))}

    subsampling = {
        'image_subsampling': torch.arange(1),
        'audio_subsampling': torch.arange(1),
        'label_subsampling': None
    }

    with torch.no_grad():
        outputs = model(**{**inputs, **subsampling})

    return outputs.logits['label']


if __name__ == '__main__':
    # select a single video
    video_names = list_ucf_videos()
    video_path = fetch_ucf_video(video_names[0])

    # extract sound from the video
    process = subprocess.Popen(
        f'!yes | ffmpeg -i "{video_path}"  -c copy  -f wav -map 0:a pcm_f32le -ar 48000 output.wav',
        shell=True,
        stdout=subprocess.PIPE)
    process.wait()

    # load the audio file
    sample_rate, audio = scipy.io.wavfile.read("output.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 2**15
    elif audio.dtype != np.float32:
        raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

    # load the video
    video = load_video(video_path)

    # define the model and configure the training
    model_config = PerceiverConfig.from_pretrained('deepmind/multimodal-perceiver')
    model = IPUPerceiverForMultimodalAutoencoding(config=model_config)
    opts = poptorch.Options()

    # set precision
    if PRECISION == 'fp16':
        opts.Precision.setPartialsType(torch.float16)
        model = model.half()

    # enable profiling
    if PROFILE_DIR:
        engine_options = {
            "autoReport.all": "true",
            "debug.allowOutOfMemory": "true",
            "autoReport.directory": PROFILE_DIR,
            "opt.useAutoloader": "true",
            "target.syncReplicasIndependently": "true",
        }
        opts._Popart.set("engineOptions", engine_options)

    # cache executable
    if EXECUTABLE_CACHE_DIR:
        opts.enableExecutableCaching(EXECUTABLE_CACHE_DIR)

    model = poptorch.inferenceModel(model.eval(), options=opts)

    # Auto-encode the first 16 frames of
    # the video and one of the audio channels
    predictions = autoencode_video(
        model,
        video[None, :16],
        audio[None, :16 * AUDIO_SAMPLES_PER_FRAME, 0:1]
    )

    # Print top 5 predicted labels
    scores, indices = torch.topk(torch.softmax(predictions, dim=1), k=5)
    for score, index in zip(scores[0], indices[0]):
        print("%s: %s" % (model.config.id2label[index.item()], score.item()))
