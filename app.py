#!/usr/bin/env python

from __future__ import annotations

import argparse
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
DESCRIPTION = 'This is a demo for https://github.com/atksh/onnx-facial-lmk-detector.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


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
                                                   use_auth_token=TOKEN)
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


def main():
    args = parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 8
    sess = ort.InferenceSession('onnx-facial-lmk-detector/model.onnx',
                                sess_options=options,
                                providers=['CPUExecutionProvider'])

    func = functools.partial(run, sess=sess)
    func = functools.update_wrapper(func, run)

    image_paths = load_sample_images()
    examples = ['onnx-facial-lmk-detector/input.jpg'
                ] + [[path.as_posix()] for path in image_paths]

    gr.Interface(
        func,
        gr.inputs.Image(type='numpy', label='Input'),
        [
            gr.outputs.Image(type='numpy', label='Output'),
            gr.outputs.Carousel(gr.outputs.Image(type='numpy'),
                                label='Aligned Faces'),
        ],
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
