#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import tarfile

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as ort

TITLE = 'atksh/onnx-facial-lmk-detector'
DESCRIPTION = 'This is an unofficial demo for https://github.com/atksh/onnx-facial-lmk-detector.'

HF_TOKEN = os.getenv('HF_TOKEN')


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['001.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=HF_TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def run(image: np.ndarray, sess: ort.InferenceSession) -> np.ndarray:
    # float32, int, int, uint8, int, float32
    # (N,), (N, 4), (N, 5, 2), (N, 224, 224, 3), (N, 106, 2), (N, 2, 3)
    scores, bboxes, keypoints, aligned_images, landmarks, affine_matrices = sess.run(
        None, {'input': image[:, :, ::-1]})

    res = image[:, :, ::-1].copy()
    for box in bboxes:
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)
    for pts in landmarks:
        for pt in pts:
            cv2.circle(res, tuple(pt), 1, (255, 255, 0), cv2.FILLED)

    return res[:, :, ::-1], [face[:, :, ::-1] for face in aligned_images]


options = ort.SessionOptions()
options.intra_op_num_threads = 8
options.inter_op_num_threads = 8
sess = ort.InferenceSession('onnx-facial-lmk-detector/model.onnx',
                            sess_options=options,
                            providers=['CPUExecutionProvider'])

func = functools.partial(run, sess=sess)

image_paths = load_sample_images()
examples = [['onnx-facial-lmk-detector/input.jpg']] + [[path.as_posix()]
                                                       for path in image_paths]

gr.Interface(
    fn=func,
    inputs=gr.Image(label='Input', type='numpy'),
    outputs=[
        gr.Image(label='Output', type='numpy'),
        gr.Gallery(label='Aligned Faces', type='numpy'),
    ],
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch(show_api=False)
