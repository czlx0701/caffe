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

// {{{ ReorderLayerCOnly

template <typename Dtype>
ReorderLayerCOnly<Dtype>::~ReorderLayerCOnly() {
    delete position_ptr;
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>*) {
  const ReorderParameter& reorder_param =
      this->layer_param_.reorder_param();
  height = bottom[0]->height();
  width  = bottom[0]->width();
  size   = height * width;
  if (!position_ptr) delete position_ptr;
  position_ptr = new SyncedMemory(sizeof(int) * size);
  int *positions = reinterpret_cast<int *>(position_ptr->mutable_cpu_data());
  if (reorder_param.position_size()) {
      CHECK_EQ(reorder_param.row_size(), 0);
      CHECK_EQ(reorder_param.col_size(), 0);
      CHECK_EQ(reorder_param.position_size(), size);
      for (int i = 0; i < size; i++) {
          int position = reorder_param.position(i);
          CHECK_LT(position, size) <<
          "Position should be less than width * height.";
          CHECK_GE(position, 0) <<
          "Position should be larger than or equal to 0.";
          positions[i] = position;
      }
  } else {
      CHECK_EQ(reorder_param.row_size(), reorder_param.col_size());
      CHECK_EQ(reorder_param.row_size(), size);
      for (int i = 0; i < reorder_param.row_size(); i++) {
          int row = reorder_param.row(i);
          int col = reorder_param.col(i);
          CHECK_LT(row, height);
          CHECK_LT(col, width);
          positions[i] = row * width + col;
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
        bottom[0]->channels() * height * width, 1, 1);
  CHECK_EQ(this->count_, (*top)[0]->count());
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Blob<Dtype>* bsrc = bottom[0];
  Blob<Dtype>* bdst = (*top)[0];
  const Dtype* src  = bsrc->cpu_data();
  const int* positions = reinterpret_cast<const int*>(position_ptr->cpu_data());
  Dtype* dst  = bdst->mutable_cpu_data();
  int channel = bsrc->channels();
  for (int n = 0; n < bsrc->num(); n++) {
      int base  = n * channel * height * width;
      for (int i = 0; i < size; i++) {
          int position = positions[i];
          for (int c = 0; c < channel; c++) {
              int index_src = base + c * height * width + position;
              int index_dst = base + channel * i + c;
              dst[index_dst] = src[index_src];
              CHECK_LT(index_src, this->count_);
              CHECK_LT(index_src, this->count_);
          }
      }
  }
}

template <typename Dtype>
void ReorderLayerCOnly<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Blob<Dtype>* bsrc = (*bottom)[0];
  Blob<Dtype>* bdst = top[0];
  Dtype*       src  = bsrc->mutable_cpu_diff();
  const Dtype* dst  = bdst->cpu_diff();
  const int* positions = reinterpret_cast<const int*>(position_ptr->cpu_data());
  int channel = bsrc->channels();
  for (int n = 0; n < bsrc->num(); n++) {
      int base  = n * channel * height * width;
      for (int i = 0; i < size; i++) {
          int position = positions[i];
          for (int c = 0; c < channel; c++) {
              int index_src = base + c * height * width + position;
              int index_dst = base + channel * i + c;
              src[index_src] = dst[index_dst];
              CHECK_LT(index_src, this->count_);
              CHECK_LT(index_src, this->count_);
          }
      }
  }
}

// }}}

// {{{ ReorderLayerCHW
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

template <typename Dtype>
void ReorderLayerCHW<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) { return; }
  Blob<Dtype>* bsrc = (*bottom)[0];
  const Blob<Dtype>* bdst = top[0];
  Dtype* src = bsrc->mutable_cpu_diff();
  const Dtype* dst = bdst->cpu_diff();
  for (int n = 0; n < num_; n++) {
      int base = n * channels_ * height_ * width_;
      for (int c = 0; c < channels_; c++) {
          for (int h = 0; h < height_; h++) {
              for (int w = 0; w < width_; w++) {
                  int index_src = base + (h * width_ + w) * channels_ + c;
                  int index_dst = base + (c * height_ + h) * width_ + w;
                  CHECK_LT(index_src, this->count_);
                  CHECK_LT(index_dst, this->count_);
                  src[index_src] = dst[index_dst];
              }
          }
      }
  }
}
// }}}

#ifdef CPU_ONLY
STUB_GPU(ReorderLayerCOnly);
STUB_GPU(ReorderLayerCHW);
#endif

INSTANTIATE_CLASS(ReorderLayer);
INSTANTIATE_CLASS(ReorderLayerCHW);
INSTANTIATE_CLASS(ReorderLayerCOnly);

}  // namespace caffe

