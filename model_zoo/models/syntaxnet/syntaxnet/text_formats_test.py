# coding=utf-8
# Copyright 2016 Google Inc. All Rights Reserved.
#
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

"""Tests for english_tokenizer."""


# disable=no-name-in-module,unused-import,g-bad-import-order,maybe-no-member
import os.path
import tensorflow as tf

import syntaxnet.load_parser_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging

from syntaxnet import sentence_pb2
from syntaxnet import task_spec_pb2
from syntaxnet.ops import gen_parser_ops

FLAGS = tf.app.flags.FLAGS


class TextFormatsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    if not hasattr(FLAGS, 'test_srcdir'):
      FLAGS.test_srcdir = ''
    if not hasattr(FLAGS, 'test_tmpdir'):
      FLAGS.test_tmpdir = tf.test.get_temp_dir()
    self.corpus_file = os.path.join(FLAGS.test_tmpdir, 'documents.conll')
    self.context_file = os.path.join(FLAGS.test_tmpdir, 'context.pbtxt')

  def AddInput(self, name, file_pattern, record_format, context):
    inp = context.input.add()
    inp.name = name
    inp.record_format.append(record_format)
    inp.part.add().file_pattern = file_pattern

  def AddParameter(self, name, value, context):
    param = context.parameter.add()
    param.name = name
    param.value = value

  def WriteContext(self, corpus_format):
    context = task_spec_pb2.TaskSpec()
    self.AddInput('documents', self.corpus_file, corpus_format, context)
    for name in ('word-map', 'lcword-map', 'tag-map',
                 'category-map', 'label-map', 'prefix-table',
                 'suffix-table', 'tag-to-category'):
      self.AddInput(name, os.path.join(FLAGS.test_tmpdir, name), '', context)
    logging.info('Writing context to: %s', self.context_file)
    with open(self.context_file, 'w') as f:
      f.write(str(context))

  def ReadNextDocument(self, sess, sentence):
    sentence_str, = sess.run([sentence])
    if sentence_str:
      sentence_doc = sentence_pb2.Sentence()
      sentence_doc.ParseFromString(sentence_str[0])
    else:
      sentence_doc = None
    return sentence_doc

  def CheckTokenization(self, sentence, tokenization):
    self.WriteContext('english-text')
    logging.info('Writing text file to: %s', self.corpus_file)
    with open(self.corpus_file, 'w') as f:
      f.write(sentence)
    sentence, _ = gen_parser_ops.document_source(
        self.context_file, batch_size=1)
    with self.test_session() as sess:
      sentence_doc = self.ReadNextDocument(sess, sentence)
      self.assertEqual(' '.join([t.word for t in sentence_doc.token]),
                       tokenization)

  def CheckUntokenizedDoc(self, sentence, words, starts, ends):
    self.WriteContext('untokenized-text')
    logging.info('Writing text file to: %s', self.corpus_file)
    with open(self.corpus_file, 'w') as f:
      f.write(sentence)
    sentence, _ = gen_parser_ops.document_source(
        self.context_file, batch_size=1)
    with self.test_session() as sess:
      sentence_doc = self.ReadNextDocument(sess, sentence)
      self.assertEqual(len(sentence_doc.token), len(words))
      self.assertEqual(len(sentence_doc.token), len(starts))
      self.assertEqual(len(sentence_doc.token), len(ends))
      for i, token in enumerate(sentence_doc.token):
        self.assertEqual(token.word.encode('utf-8'), words[i])
        self.assertEqual(token.start, starts[i])
        self.assertEqual(token.end, ends[i])

  def testUntokenized(self):
    self.CheckUntokenizedDoc('????????????', ['???', '???', '???', '???'],
                             [0, 3, 6, 9], [2, 5, 8, 11])
    self.CheckUntokenizedDoc('Hello ', ['H', 'e', 'l', 'l', 'o', ' '],
                             [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

  def testSegmentationTrainingData(self):
    doc1_lines = ['??????	NO_SPACE\n',
                  '???	NO_SPACE\n',
                  '??????	NO_SPACE']
    doc1_text = '???????????????'
    doc1_tokens = ['???', '???', '???', '???', '???']
    doc1_break_levles = [1, 0, 1, 1, 0]
    doc2_lines = ['That	NO_SPACE\n',
                  '\'s	SPACE\n',
                  'a	SPACE\n',
                  'good	SPACE\n',
                  'point	NO_SPACE\n',
                  '.	NO_SPACE']
    doc2_text = 'That\'s a good point.'
    doc2_tokens = ['T', 'h', 'a', 't', '\'', 's', ' ', 'a', ' ', 'g', 'o', 'o',
                   'd', ' ', 'p', 'o', 'i', 'n', 't', '.']
    doc2_break_levles = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                         0, 1]
    self.CheckSegmentationTrainingData(doc1_lines, doc1_text, doc1_tokens,
                                       doc1_break_levles)
    self.CheckSegmentationTrainingData(doc2_lines, doc2_text, doc2_tokens,
                                       doc2_break_levles)

  def CheckSegmentationTrainingData(self, doc_lines, doc_text, doc_words,
                                    break_levels):
    # Prepare context.
    self.WriteContext('segment-train-data')

    # Prepare test sentence.
    with open(self.corpus_file, 'w') as f:
      f.write(''.join(doc_lines))

    # Test converted sentence.
    sentence, _ = gen_parser_ops.document_source(
        self.context_file, batch_size=1)
    with self.test_session() as sess:
      sentence_doc = self.ReadNextDocument(sess, sentence)
      self.assertEqual(doc_text.decode('utf-8'), sentence_doc.text)
      self.assertEqual([t.decode('utf-8') for t in doc_words],
                       [t.word for t in sentence_doc.token])
      self.assertEqual(break_levels,
                       [t.break_level for t in sentence_doc.token])

  def testSimple(self):
    self.CheckTokenization('Hello, world!', 'Hello , world !')
    self.CheckTokenization('"Hello"', "`` Hello ''")
    self.CheckTokenization('{"Hello@#$', '-LRB- `` Hello @ # $')
    self.CheckTokenization('"Hello..."', "`` Hello ... ''")
    self.CheckTokenization('()[]{}<>',
                           '-LRB- -RRB- -LRB- -RRB- -LRB- -RRB- < >')
    self.CheckTokenization('Hello--world', 'Hello -- world')
    self.CheckTokenization("Isn't", "Is n't")
    self.CheckTokenization("n't", "n't")
    self.CheckTokenization('Hello Mr. Smith.', 'Hello Mr. Smith .')
    self.CheckTokenization("It's Mr. Smith's.", "It 's Mr. Smith 's .")
    self.CheckTokenization("It's the Smiths'.", "It 's the Smiths ' .")
    self.CheckTokenization('Gotta go', 'Got ta go')
    self.CheckTokenization('50-year-old', '50-year-old')

  def testUrl(self):
    self.CheckTokenization('http://www.google.com/news is down',
                           'http : //www.google.com/news is down')


if __name__ == '__main__':
  googletest.main()
