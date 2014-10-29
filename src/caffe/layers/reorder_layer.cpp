#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
ReorderLayer<Dtype>* ReorderLayer<Dtype>::Create(
    const LayerParameter& param) {
    const ReorderParameter& reorder_param = param.reorder_param();
    switch (reorder_param.order()) {
        case ReorderParameter_StorageOrder_ChannelHeightWeight:
            return new ReorderLayerCHW<Dtype>(param);
            break;
        case ReorderParameter_StorageOrder_ChannelOnly:
            return new ReorderLayerCOnly<Dtype>(param);
            break;
    }
    LOG(FATAL) << "Undefined reorder type:" << reorder_param.order();
    return NULL;
}

template <typename Dtype>
void ReorderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  caffe_copy(count_, top[0]->cpu_diff(), (*bottom)[0]->mutable_cpu_diff());
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const ReorderParameter& reorder_param =
      this->layer_param_.reorder_param();
  position_.clear();
  CHECK_EQ(reorder_param.row_size(), reorder_param.col_size());
  CHECK_EQ(reorder_param.row_size(),
    bottom[0]->height() * bottom[1]->width());
  for (int i = 0; i < reorder_param.row_size(); i++) {
      int row = reorder_param.row(i);
      int col = reorder_param.col(i);
      position_.push_back(make_pair(row, col));
  }
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  this->count_ = bottom[0]->count();
  CHECK_NE((*top)[0], bottom[0]) << this->type_name() << " Layer does not "
        "allow in-place computation.";
  (*top)[0]->Reshape(bottom[0]->num(),
        bottom[0]->channels() * bottom[0]->height() * bottom[0]->width(),
        1, 1);
  CHECK_EQ(this->count_, (*top)[0]->count());
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Blob<Dtype>* bsrc = bottom[0];
  Blob<Dtype>* bdst = (*top)[0];
  const Dtype* src = bsrc->cpu_data();
  Dtype* dst = bdst->mutable_cpu_data();
  int index = 0;
  int num   = bsrc->num();
  for (vector<Coord>::const_iterator iter = position_.begin();
          iter != position_.end(); ++iter) {
      int row = iter->first;
      int col = iter->first;
      for (int c = 0; c < bsrc->channels(); c++) {
          int index_src = bsrc->offset(num, c, row, col);
          dst[index] = src[index_src];
          index++;
      }
  }
  CHECK_EQ(index, this->count_);
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(ReorderLayer, Backward);
STUB_GPU_FORWARD(ReorderLayerCHW, Forward);
STUB_GPU_FORWARD(ReorderLayerCOnly, Forward);
#endif

INSTANTIATE_CLASS(ReorderLayer);
INSTANTIATE_CLASS(ReorderLayerCHW);
INSTANTIATE_CLASS(ReorderLayerCOnly);

}  // namespace caffe

