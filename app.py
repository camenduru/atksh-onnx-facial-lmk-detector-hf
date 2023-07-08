#!/usr/bin/env python

from __future__ import annotations

import functools
import os

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

DESCRIPTION = '# [atksh/onnx-facial-lmk-detector](https://github.com/atksh/onnx-facial-lmk-detector)'


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

fn = functools.partial(run, sess=sess)

examples = [['onnx-facial-lmk-detector/input.jpg'],
            ['images/pexels-ksenia-chernaya-8535230.jpg']]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input', type='numpy')
            run_button = gr.Button()
        with gr.Column():
            result = gr.Image(label='Output')
            gallery = gr.Gallery(label='Aligned Faces')
    gr.Examples(examples=examples,
                inputs=image,
                outputs=[result, gallery],
                fn=fn,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')
    run_button.click(fn=fn,
                     inputs=image,
                     outputs=[result, gallery],
                     api_name='run')
demo.queue(max_size=10).launch()
