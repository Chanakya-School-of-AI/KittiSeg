"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

# Tensorflow 1.14 compatibility - Vipul Vaibhaw

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

# Vipul - Importing libraries because scipy.misc image are removed!
import cv2 
from PIL import Image

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp

import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return
      
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image
    image = cv2.imread(input_image)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = cv2.resize(image, (image_width, image_height),
                                  interpolation='cubic')

    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']
    output = sess.run([softmax], feed_dict=feed)

    # Reshape output from flat vector to 2D Image
    height, width, _  = image.shape
    output_image = output[0][:, 1].reshape(height, width)

    # Plot confidences as red-blue overlay
    rb_image = seg.make_overlay(image, output_image)

    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
    threshold = 0.5
    street_prediction = output_image > threshold

    # Plot the hard prediction as green overlay
    # green_image = tv_utils.fast_overlay(image, street_prediction)
    # The above logic is obsolete code. Following code is cv2 logic of above code.

    # Background img
    green_image_background = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    b, g, r, alpha = cv2.split(green_image_background)
    green_image_background[:,:, 0] = image[:,:,0]
    green_image_background[:,:, 1] = image[:,:,1]
    green_image_background[:,:, 2] = image[:,:,2]
    green_image_background[:,:, 3] = alpha[:,:]

    # forground mask of road
    color = [0, 255, 0, 127]
    color = np.array(color).reshape(1, 4)
    shape = image.shape
    segmentation = street_prediction.reshape(shape[0], shape[1], 1)

    output = np.dot(segmentation, color).astype('uint8') 

    # Combine both background and foreground
    green_image = cv2.addWeighted(green_image_background,1,output,0.4,0)
    green_image = cv2.cvtColor(green_image,cv2.COLOR_RGBA2BGR)
    
    # Save output images to disk.
    if FLAGS.output_image is None:
        output_base_name = input_image
    else:
        output_base_name = FLAGS.output_image

    raw_image_name = output_base_name.split('.')[0] + '_raw.png'
    rb_image_name = output_base_name.split('.')[0] + '_rb.png'
    green_image_name = output_base_name.split('.')[0] + '_green.png'

    cv2.imwrite(raw_image_name, output_image)
    cv2.imwrite(rb_image_name, rb_image)
    cv2.imwrite(green_image_name, green_image)
    cv2.imshow(green_image_name, green_image)
    cv2.waitKey(5000)

    logging.info("")
    logging.info("Raw output image has been saved to: {}".format(
        os.path.realpath(raw_image_name)))
    logging.info("Red-Blue overlay of confs have been saved to: {}".format(
        os.path.realpath(rb_image_name)))
    logging.info("Green plot of predictions have been saved to: {}".format(
        os.path.realpath(green_image_name)))

    logging.info("")
    logging.warning("Do NOT use this Code to evaluate multiple images.")

    logging.warning("Demo.py is **very slow** and designed "
                    "to be a tutorial to show how the KittiSeg works.")
    logging.warning("")
    logging.warning("Please see this comment, if you like to apply demo.py to"
                    "multiple images see:")
    logging.warning("https://github.com/MarvinTeichmann/KittiBox/"
                    "issues/15#issuecomment-301800058")


if __name__ == '__main__':
    tf.app.run()
