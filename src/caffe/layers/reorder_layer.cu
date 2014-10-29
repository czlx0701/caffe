#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {
/*
template <typename Dtype>
void ReorderLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    (*top)[i]->ShareData(*bottom[0]);
  }
}
*/

template <typename Dtype>
void ReorderLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  caffe_copy(count_, top[0]->gpu_diff(), (*bottom)[0]->mutable_gpu_diff());
}

INSTANTIATE_CLASS(ReorderLayer);

}  // namespace caffe
