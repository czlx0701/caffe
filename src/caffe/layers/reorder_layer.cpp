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
      vector<Blob<Dtype>*>*) {
  const ReorderParameter& reorder_param =
      this->layer_param_.reorder_param();
  position_.clear();
  if (reorder_param.position_size()) {
      int size = bottom[0]->height() * bottom[0]->width();
      CHECK_EQ(reorder_param.row_size(), 0);
      CHECK_EQ(reorder_param.col_size(), 0);
      CHECK_EQ(reorder_param.position_size(), size);
      for (int i = 0; i < reorder_param.position_size(); i++) {
          int position = reorder_param.position(i);
          CHECK_LT(position, size) <<
          "Position should be less than width * height.";
          CHECK_GE(position, 0) <<
          "Position should be larger than or equal to 0.";
          position_.push_back(position);
      }
  } else {
      CHECK_EQ(reorder_param.row_size(), reorder_param.col_size());
      CHECK_EQ(reorder_param.row_size(),
              bottom[0]->height() * bottom[0]->width());
      int height = bottom[0]->height();
      int width  = bottom[0]->width();
      for (int i = 0; i < reorder_param.row_size(); i++) {
          int row = reorder_param.row(i);
          int col = reorder_param.col(i);
          CHECK_LT(row, height);
          CHECK_LT(col, width);
          position_.push_back(row * width + col);
      }
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
  Dtype* dst  = bdst->mutable_cpu_data();
  int channel = bsrc->channels();
  int height  = bsrc->height();
  int width   = bsrc->width();
  int index   = 0;
  int base    = 0;
  for (int n = 0; n < bsrc->num(); n++) {
      base  = n * channel * height * width;
      index = 0;
      for (vector<int>::const_iterator iter = position_.begin();
              iter != position_.end(); ++iter) {
          int position = *iter;
          for (int c = 0; c < channel; c++) {
              int index_src = base + c * height * width + position;
              dst[index + base] = src[index_src];
              CHECK_LT(index_src, this->count_);
              CHECK_LT(index + base, this->count_);
              index++;
          }
      }
  }
  CHECK_EQ(index + base, this->count_);
}

template <typename Dtype>
void ReorderLayerCHW<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>*) {
  const ReorderParameter& reorder_param =
      this->layer_param_.reorder_param();
  num_ = bottom[0]->num();
#define REQUIRE_SET_PARAM(name)     \
  CHECK(reorder_param.has_##name()) << #name " missing.";   \
  name##_ = reorder_param.name();
  REQUIRE_SET_PARAM(channels);
  REQUIRE_SET_PARAM(height);
  REQUIRE_SET_PARAM(width);
#undef REQUIRE_SET_PARAM
  CHECK_EQ(num_ * channels_ * height_ * width_, bottom[0]->count());
}

template <typename Dtype>
void ReorderLayerCHW<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  this->count_ = bottom[0]->count();
  CHECK_NE((*top)[0], bottom[0]) << this->type_name() << " Layer does not "
        "allow in-place computation.";
  // (*top)[0]->Reshape(bottom[0]->num(), channels_ * height_ * width_, 1, 1);
  (*top)[0]->ReshapeLike(*(bottom[0]));
  CHECK_EQ(this->count_, (*top)[0]->count());
}

template <typename Dtype>
void ReorderLayerCHW<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Blob<Dtype>* bsrc = bottom[0];
  Blob<Dtype>* bdst = (*top)[0];
  const Dtype* src = bsrc->cpu_data();
  Dtype* dst = bdst->mutable_cpu_data();
  for (int n = 0; n < num_; n++) {
      int base = n * channels_ * height_ * width_;
      for (int c = 0; c < channels_; c++) {
          for (int h = 0; h < height_; h++) {
              for (int w = 0; w < width_; w++) {
                  int index_src = base + (h * width_ + w) * channels_ + c;
                  int index_dst = base + (c * height_ + h) * width_ + w;
                  CHECK_LT(index_src, this->count_);
                  CHECK_LT(index_dst, this->count_);
                  dst[index_dst] = src[index_src];
              }
          }
      }
  }
}

INSTANTIATE_CLASS(ReorderLayer);
INSTANTIATE_CLASS(ReorderLayerCHW);
INSTANTIATE_CLASS(ReorderLayerCOnly);

}  // namespace caffe

