#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Neural Network Image Compression Decoder.

Decompress an image from the numpy's npz format generated by the encoder.

Example usage:
python decoder.py --input_codes=output_codes.pkl --iteration=15 \
--output_directory=/tmp/compression_output/ --model=residual_gru.pb
"""
import io
import os

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string('input_codes', None, 'Location of binary code file.')
tf.flags.DEFINE_integer('iteration', -1, 'The max quality level of '
                        'the images to output. Use -1 to infer from loaded '
                        ' codes.')
tf.flags.DEFINE_string('output_directory', None, 'Directory to save decoded '
                       'images.')
tf.flags.DEFINE_string('model', None, 'Location of compression model.')

FLAGS = tf.flags.FLAGS


def get_input_tensor_names():
  name_list = ['GruBinarizer/SignBinarizer/Sign:0']
  for i in xrange(1, 16):
    name_list.append('GruBinarizer/SignBinarizer/Sign_{}:0'.format(i))
  return name_list


def get_output_tensor_names():
  return ['loop_{0:02d}/add:0'.format(i) for i in xrange(0, 16)]


def main(_):
  if (FLAGS.input_codes is None or FLAGS.output_directory is None or
      FLAGS.model is None):
    print ('\nUsage: python decoder.py --input_codes=output_codes.pkl '
           '--iteration=15 --output_directory=/tmp/compression_output/ '
           '--model=residual_gru.pb\n\n')
    return

  if FLAGS.iteration < -1 or FLAGS.iteration > 15:
    print ('\n--iteration must be between 0 and 15 inclusive, or -1 to infer '
           'from file.\n')
    return
  iteration = FLAGS.iteration

  if not tf.gfile.Exists(FLAGS.output_directory):
    tf.gfile.MkDir(FLAGS.output_directory)

  if not tf.gfile.Exists(FLAGS.input_codes):
    print '\nInput codes not found.\n'
    return

  contents = ''
  with tf.gfile.FastGFile(FLAGS.input_codes, 'r') as code_file:
    contents = code_file.read()
    loaded_codes = np.load(io.BytesIO(contents))
    assert ['codes', 'shape'] not in loaded_codes.files
    loaded_shape = loaded_codes['shape']
    loaded_array = loaded_codes['codes']

    # Unpack and recover code shapes.
    unpacked_codes = np.reshape(np.unpackbits(loaded_array)
                                [:np.prod(loaded_shape)],
                                loaded_shape)

    numpy_int_codes = np.split(unpacked_codes, len(unpacked_codes))
    if iteration == -1:
      iteration = len(unpacked_codes) - 1
    # Convert back to float and recover scale.
    numpy_codes = [np.squeeze(x.astype(np.float32), 0) * 2 - 1 for x in
                   numpy_int_codes]

  with tf.Graph().as_default() as graph:
    # Load the inference model for decoding.
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    _ = tf.import_graph_def(graph_def, name='')

    # For encoding the tensors into PNGs.
    input_image = tf.placeholder(tf.uint8)
    encoded_image = tf.image.encode_png(input_image)

    input_tensors = [graph.get_tensor_by_name(name) for name in
                     get_input_tensor_names()][0:iteration+1]
    outputs = [graph.get_tensor_by_name(name) for name in
               get_output_tensor_names()][0:iteration+1]

  feed_dict = {key: value for (key, value) in zip(input_tensors,
                                                  numpy_codes)}

  with tf.Session(graph=graph) as sess:
    results = sess.run(outputs, feed_dict=feed_dict)

    for index, result in enumerate(results):
      img = np.uint8(np.clip(result + 0.5, 0, 255))
      img = img.squeeze()
      png_img = sess.run(encoded_image, feed_dict={input_image: img})

      with tf.gfile.FastGFile(os.path.join(FLAGS.output_directory,
                                           'image_{0:02d}.png'.format(index)),
                              'w') as output_image:
        output_image.write(png_img)


if __name__ == '__main__':
  tf.app.run()
