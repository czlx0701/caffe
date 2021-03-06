#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReorderCOnlyForward(const int num, const int channels,
        const int height, const int width, const int* positions,
        const Dtype* in, Dtype* out) {
    for (int n = 0; n < num; n++) {
        int base = n * channels * height * width;
        for (int i = 0; i < height * width; i++) {
            int position = positions[i];
            CUDA_KERNEL_LOOP(c, channels) {
                int index_src = base + c * height * width + position;
                int index_dst = base + channels * i + c;
                out[index_dst] = in[index_src];
            }
        }
    }
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data    = (*top)[0]->mutable_gpu_data();
  const int num      = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int *pos = reinterpret_cast<const int *>(position_ptr->cpu_data());
  const int *positions =
      reinterpret_cast<const int *>(position_ptr->gpu_data());
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(num * channels * height * width, (*top)[0]->count());
  CHECK_EQ(num * channels * height * width, bottom[0]->count());
  // NOLINT_NEXT_LINE(*)
  ReorderCOnlyForward<Dtype><<<CAFFE_GET_BLOCKS(channels), CAFFE_CUDA_NUM_THREADS>>>(
      num, channels, height, width, positions, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReorderCOnlyBackward(const int num, const int channels,
        const int height, const int width, const int* positions,
        Dtype* in, const Dtype* out) {
    for (int n = 0; n < num; n++) {
        int base = n * channels * height * width;
        for (int i = 0; i < height * width; i++) {
            int position = positions[i];
            CUDA_KERNEL_LOOP(c, channels) {
                int index_src = base + c * height * width + position;
                int index_dst = base + channels * i + c;
                in[index_src] = out[index_dst];
            }
        }
    }
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_data    = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  const int num      = (*bottom)[0]->num();
  const int channels = (*bottom)[0]->channels();
  const int *positions =
      reinterpret_cast<const int *>(position_ptr->gpu_data());
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(num * channels * height * width, top[0]->count());
  CHECK_EQ(num * channels * height * width, (*bottom)[0]->count());
  // NOLINT_NEXT_LINE(*)
  ReorderCOnlyBackward<Dtype><<<CAFFE_GET_BLOCKS(channels), CAFFE_CUDA_NUM_THREADS>>>(
      num, channels, height, width, positions, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReorderCHWForward(const int num, const int channels,
        const int height, const int width, const Dtype* in, Dtype* out) {
    for (int n = 0; n < num; n++) {
        int base = n * channels * height * width;
        CUDA_KERNEL_LOOP(c, channels) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index_src = base + (h * width + w) * channels + c;
                    int index_dst = base + (c * height + h) * width + w;
                    out[index_dst] = in[index_src];
                }
            }
        }
    }
}

template <typename Dtype>
void ReorderLayerCHW<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data    = (*top)[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(num_ * channels_ * height_ * width_, (*top)[0]->count());
  CHECK_EQ(num_ * channels_ * height_ * width_, bottom[0]->count());
  // NOLINT_NEXT_LINE(*)
  ReorderCHWForward<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_, width_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReorderCHWBackward(const int num, const int channels,
        const int height, const int width, Dtype* in, const Dtype* out) {
    for (int n = 0; n < num; n++) {
        int base = n * channels * height * width;
        CUDA_KERNEL_LOOP(c, channels) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index_src = base + (h * width + w) * channels + c;
                    int index_dst = base + (c * height + h) * width + w;
                    in[index_src] = out[index_dst];
                }
            }
        }
    }
}

template <typename Dtype>
void ReorderLayerCHW<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_data    = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(num_ * channels_ * height_ * width_, top[0]->count());
  CHECK_EQ(num_ * channels_ * height_ * width_, (*bottom)[0]->count());
  // NOLINT_NEXT_LINE(*)
  ReorderCHWBackward<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_, width_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(ReorderLayerCOnly);
INSTANTIATE_CLASS(ReorderLayerCHW);

}  // namespace caffe
