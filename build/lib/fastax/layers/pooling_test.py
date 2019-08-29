# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for conv layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from fastax.layers import base
from fastax.layers import pooling


class PoolingLayerTest(absltest.TestCase):

  def test_avg_pool(self):
    input_shape = (29, 4, 4, 20)
    result_shape = base.check_shape_agreement(
        pooling.AvgPool(pool_size=(2, 2), strides=(2, 2)), input_shape)
    self.assertEqual(result_shape, (29, 2, 2, 20))


if __name__ == "__main__":
  absltest.main()
