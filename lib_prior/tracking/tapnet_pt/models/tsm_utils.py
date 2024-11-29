# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils functions for TSM."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def prepare_inputs(inputs: chex.Array) -> Tuple[jnp.ndarray, str, int]:
  """Deduces input mode for TSM."""
  # Deduce if we run on TPU based on input shape.
  if len(inputs.shape) == 5:
    # Input is given in the standard [B, T, H, W, 3] format.
    tsm_mode = 'gpu'
    num_frames = inputs.shape[1]
    inputs = jnp.reshape(inputs, [-1] + list(inputs.shape[2:]))
  else:
    # Input is given in the [T * B, H, W, 3] format.
    tsm_mode = 'tpu'
    num_frames = None
  return inputs, tsm_mode, num_frames


def prepare_outputs(
    outputs: chex.Array,
    tsm_mode: str,
    num_frames: int,
    reduce_mean: bool = True,
) -> jnp.ndarray:
  """Processes output of TSM to undo the merging of batch and time."""
  # Get the shape without the batch/time dimension (for TSM batch and time are
  # merged in the first dimension).
  shape_no_bt = list(outputs.shape[1:])
  if tsm_mode == 'tpu':
    # Outputs are of the shape [num_frames * B, ..., n_channels]
    outputs = jnp.reshape(outputs, [num_frames, -1] + shape_no_bt)
    if reduce_mean:
      # We average over time and space.
      outputs = jnp.mean(
          outputs, axis=[0] + list(range(2,
                                         len(shape_no_bt) + 1)))
    else:
      outputs = jnp.transpose(
          outputs, axes=[1, 0] + list(range(2,
                                            len(shape_no_bt) + 2)))
  elif tsm_mode == 'gpu':
    # Outputs are of the shape [B * num_frames, ..., n_channels].
    outputs = jnp.reshape(outputs, [-1, num_frames] + shape_no_bt)
    if reduce_mean:
      outputs = jnp.mean(
          outputs, axis=[1] + list(range(2,
                                         len(shape_no_bt) + 1)))
  elif tsm_mode.startswith('deflated'):
    # In deflated mode, outputs are already in the right format.
    pass
  else:
    raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                     f'\'deflated_0.x\' ({tsm_mode} given)')
  return outputs  # pytype: disable=bad-return-type  # numpy-scalars


def apply_temporal_shift(
    x: chex.Array,
    tsm_mode: str,
    num_frames: int,
    channel_shift_fraction: float = 0.125,
) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383 with mode."""
  if tsm_mode == 'tpu':
    outputs = temporal_shift_tpu(x, num_frames, channel_shift_fraction)
  elif tsm_mode == 'gpu':
    outputs = temporal_shift_gpu(x, num_frames, channel_shift_fraction)
  elif tsm_mode.startswith('deflated'):
    alpha = float(tsm_mode.split('_')[1])
    outputs = temporal_shift_image_mode(x, channel_shift_fraction, alpha)
  else:
    raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                     f'\'deflated_0.x\' ({tsm_mode} given)')
  return outputs


def temporal_shift_image_mode(x, channel_shift_fraction=0.125, alpha=0.3):
  """Temporal shift applied on single image (to emulate a fixed video)."""
  # B, H, W, C = batch_size, im_height, im_width, channels.
  # Input is (B, H, W, C).
  orig_shp = tuple(x.shape)
  n_channels = orig_shp[-1]
  n_shift = int(n_channels * channel_shift_fraction)
  # Alpha emulates the effect of the padding when using a single frame.
  shifted_backward = alpha * x[:, :, :, -n_shift:]
  shifted_forward = alpha * x[:, :, :, :n_shift]
  no_shift = x[:, :, :, n_shift:-n_shift]
  shifted_x = jnp.concatenate([shifted_backward, no_shift, shifted_forward],
                              axis=3)
  return shifted_x


def temporal_shift_gpu(
    x: chex.Array,
    num_frames: int,
    channel_shift_fraction: float = 0.125,
) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383."""
  # B, T, H, W, C = batch_size, num_frames, im_height, im_width, channels.
  # Input is (B * T, H, W, C).
  orig_shp = tuple(x.shape)
  reshaped_x = jnp.reshape(x, (-1, num_frames) + orig_shp[1:])
  n_channels = orig_shp[-1]
  n_shift = int(n_channels * channel_shift_fraction)

  new_shp = tuple(reshaped_x.shape)

  # shifted_backward = reshaped_x[:, 1:, :, :, -n_shift:].
  shifted_backward = jax.lax.slice(
      reshaped_x, (0, 1, 0, 0, new_shp[4] - n_shift),
      (new_shp[0], new_shp[1], new_shp[2], new_shp[3], new_shp[4]))
  shifted_backward_padding = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
  shifted_backward = jnp.pad(shifted_backward, shifted_backward_padding)

  # shifted_forward = reshaped_x[:, :-1, :, :, :n_shift].
  shifted_forward = jax.lax.slice(
      reshaped_x, (0, 0, 0, 0, 0),
      (new_shp[0], new_shp[1] - 1, new_shp[2], new_shp[3], n_shift))
  shifted_forward_padding = ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0))
  shifted_forward = jnp.pad(shifted_forward, shifted_forward_padding)

  no_shift = reshaped_x[:, :, :, :, n_shift:-n_shift]
  shifted_x = jnp.concatenate([shifted_backward, no_shift, shifted_forward],
                              axis=4)
  return jnp.reshape(shifted_x, (-1,) + orig_shp[1:])


def temporal_shift_tpu(
    x: chex.Array,
    num_frames: int,
    channel_shift_fraction: float = 0.125,
) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383.

    TPU optimized version of TSM. Reshape is avoided by having the images
    reshaped in [T * B, :] so that frames corresponding to same time frame in
    videos are contiguous in memory. Finally, to avoid concatenate that prevent
    some fusion from happening we simply sum masked version of the features.
  Args:
    x: Input expected to be [T * B, H, W, C] (where the batch has been reshaped
      from a time major version of the input).
    num_frames: number of frames T per video.
    channel_shift_fraction: fraction of the channel to shift forward and
      backward.

  Returns:
      The temporal shifted version of x.
  """
  # B, T, H, W, C = batch_size, num_frames, im_height, im_width, channels.
  # Input is (T * B, H, W, C).
  original_dtype = x.dtype
  original_shape = list(x.shape)

  batch_size = int(original_shape[0] / num_frames)
  n_channels = int(original_shape[-1])
  n_shift = int(n_channels * channel_shift_fraction)

  # Cast to bfloat16.
  x = x.astype(jnp.bfloat16)

  # For the following, assume that x has 3 channels [x1, x2, x3] and n_shift=1.
  # Shift backward, we first pad by zeros [x1, x2, x3, 0, 0].
  orig_shp = list(x.shape)

  shifted_backward_padding = ((0, batch_size, 0), (0, 0, 0), (0, 0, 0),
                              (0, n_channels - n_shift, 0))
  x_backward_padding = jax.lax.pad(
      x,
      padding_value=jnp.bfloat16(0.),
      padding_config=shifted_backward_padding)
  # The following shift gets to [x3^+1, 0, 0] (where +1 means from the future).
  shifted_backward = jax.lax.slice(x_backward_padding,
                                   (batch_size, 0, 0, n_channels - n_shift),
                                   (orig_shp[0] + batch_size, orig_shp[1],
                                    orig_shp[2], 2 * n_channels - n_shift))
  # Shift forward, we first pad by zeros [0, 0, x1, x2, x3].
  shifted_forward_padding = ((batch_size, 0, 0), (0, 0, 0), (0, 0, 0),
                             (n_channels - n_shift, 0, 0))
  x_forward_padding = jax.lax.pad(
      x, padding_value=jnp.bfloat16(0.), padding_config=shifted_forward_padding)
  # The following shift gets to [0, 0, x1^-1] (where -1 means from the past).
  shifted_forward = jax.lax.slice(
      x_forward_padding, (0, 0, 0, 0),
      (orig_shp[0], orig_shp[1], orig_shp[2], n_channels))
  # No shift is in the middle, this gets [0, x2, 0].
  mask_noshift = (jnp.reshape((jnp.arange(n_channels) >= n_shift) &
                              (jnp.arange(n_channels) < n_channels - n_shift),
                              (1, 1, 1, -1))).astype(jnp.bfloat16)
  no_shift = mask_noshift * x
  # By summing everything together, we end up with [x3^+1, x2, x1^-1].
  # Note: channels have been reordered but that doesn't matter for the model.
  shifted_x = shifted_backward + shifted_forward + no_shift

  return shifted_x.astype(original_dtype)
