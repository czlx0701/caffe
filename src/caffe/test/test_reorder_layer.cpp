#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// {{{ReorderLayerCOnlyTest
template <typename TypeParam>
class ReorderLayerCOnlyTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReorderLayerCOnlyTest()
      : blob_bottom_(new Blob<Dtype>(128, 256, 2, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~ReorderLayerCOnlyTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(ReorderLayerCOnlyTest, TestDtypesAndDevices);

static void add_param_pos(ReorderParameter *param)
{
#define ADD_POS(row, col) param->add_row(row); param->add_col(col);
  ADD_POS(1, 2);
  ADD_POS(0, 1);
  ADD_POS(1, 0);
  ADD_POS(0, 2);
  ADD_POS(1, 1);
  ADD_POS(0, 0);
#undef ADD_POS
}

static LayerParameter create_reorder_layer_c_param() {
  LayerParameter layer_param;
  ReorderParameter *param = layer_param.mutable_reorder_param();
  param->set_order(ReorderParameter_StorageOrder_ChannelOnly);
  return layer_param;
}

template <typename Dtype>
static ReorderLayer<Dtype>* create_reorder_layer_c() {
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  add_param_pos(param);
  return ReorderLayer<Dtype>::Create(layer_param);
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumRowCol) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer(create_reorder_layer_c<Dtype>()); 
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_top_->channels(),
          this->blob_bottom_->channels() * this->blob_bottom_->height() *
          this->blob_bottom_->width());
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumRowCol_Less) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  add_param_pos(param);
  param->add_row(1); param->add_col(0);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumRowCol_More) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  add_param_pos(param);
  param->add_row(1);
  param->add_row(0);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumRowCol_NE) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  add_param_pos(param);
  param->add_row(1);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumPos) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  param->add_position(1);
  param->add_position(2);
  param->add_position(3);
  param->add_position(4);
  param->add_position(5);
  param->add_position(0);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_top_->channels(),
          this->blob_bottom_->channels() * this->blob_bottom_->height() *
          this->blob_bottom_->width());
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumPos_Less) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  param->add_position(1);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

TYPED_TEST(ReorderLayerCOnlyTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer_conly(create_reorder_layer_c<Dtype>());
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  layer_conly->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer_conly->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  caffe_copy(this->blob_top_->count(),
        reinterpret_cast<const Dtype *>(this->blob_top_->cpu_data()),
        reinterpret_cast<Dtype *>(this->blob_top_->mutable_cpu_diff()));
  layer_conly->Backward(this->blob_top_vec_, propagate_down, &(this->blob_bottom_vec_));
  const Dtype *src = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_data());
  const Dtype *dst = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_diff());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
      EXPECT_EQ(src[i], dst[i]);
  }
}

TYPED_TEST(ReorderLayerCOnlyTest, TestSetupNumPos_More) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param = create_reorder_layer_c_param();
  ReorderParameter *param = layer_param.mutable_reorder_param();
  param->add_position(1);
  param->add_position(2);
  param->add_position(3);
  param->add_position(4);
  param->add_position(5);
  param->add_position(0);
  param->add_position(0);
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(layer_param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

// }}}

// {{{ ReorderLayerCHWTest
template <typename TypeParam>
class ReorderLayerCHWTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReorderLayerCHWTest()
      : blob_bottom_(new Blob<Dtype>(128, 256 * 2 * 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~ReorderLayerCHWTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(ReorderLayerCHWTest, TestDtypesAndDevices);

static LayerParameter create_reorder_layer_chw_param() {
  LayerParameter layer_param;
  ReorderParameter *param = layer_param.mutable_reorder_param();
  param->set_order(ReorderParameter_StorageOrder_ChannelHeightWeight);
  param->set_channels(256);
  param->set_height(2);
  param->set_width(3);
  return layer_param;
}

template <typename Dtype>
static ReorderLayer<Dtype>* create_reorder_layer_chw() {
  LayerParameter layer_param = create_reorder_layer_chw_param();
  return ReorderLayer<Dtype>::Create(layer_param);
}

TYPED_TEST(ReorderLayerCHWTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer(create_reorder_layer_chw<Dtype>()); 
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
}

TYPED_TEST(ReorderLayerCHWTest, TestSetupMissing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param = create_reorder_layer_chw_param();
  param.mutable_reorder_param()->clear_channels();
  shared_ptr< ReorderLayer<Dtype> > layer(ReorderLayer<Dtype>::Create(param)); 
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
          layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_)), "");
}

TYPED_TEST(ReorderLayerCHWTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer_chw(create_reorder_layer_chw<Dtype>());
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  layer_chw->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer_chw->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  caffe_copy(this->blob_top_->count(),
        reinterpret_cast<const Dtype *>(this->blob_top_->cpu_data()),
        reinterpret_cast<Dtype *>(this->blob_top_->mutable_cpu_diff()));
  layer_chw->Backward(this->blob_top_vec_, propagate_down, &(this->blob_bottom_vec_));
  const Dtype *src = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_data());
  const Dtype *dst = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_diff());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
      CHECK_EQ(src[i], dst[i]);
  }
}
// }}}

// {{{ReorderLayerTest
template <typename TypeParam>
class ReorderLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReorderLayerTest()
      : blob_bottom_(new Blob<Dtype>(128, 256, 2, 3)),
        blob_top_(new Blob<Dtype>()),
        blob_result_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_result_vec_.push_back(blob_result_);
  }

  virtual ~ReorderLayerTest() {
    delete blob_top_;
    delete blob_bottom_;
    delete blob_result_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_result_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_result_vec_;
};

TYPED_TEST_CASE(ReorderLayerTest, TestDtypesAndDevices);

template <typename Dtype>
static void check_reorder_result(const Blob<Dtype> &src, const Blob<Dtype> &dst) {
#define DEC_GET_VAR(name)   int name = src.name();
    DEC_GET_VAR(num);
    DEC_GET_VAR(channels);
    DEC_GET_VAR(height);
    DEC_GET_VAR(width);
#undef DEC_GET_VAR
    LayerParameter layer_param = create_reorder_layer_c_param();
    ReorderParameter &param = *layer_param.mutable_reorder_param();
    add_param_pos(&param);
    ASSERT_EQ(param.row_size(), height * width);
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channels; c++) {
            int h_dst = 0;
            int w_dst = 0;
            for (int i = 0; i < param.row_size(); i++) {
                int h_src = param.row(i);
                int w_src = param.col(i);
                EXPECT_LT(h_dst, height);
                EXPECT_LT(w_dst, width);
                EXPECT_EQ(src.data_at(n, c, h_src, w_src),
                        dst.data_at(n, c, h_dst, w_dst));
                w_dst++;
                h_dst += w_dst / width;
                w_dst %= width;
            }
        }
    }
}

TYPED_TEST(ReorderLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer_conly(create_reorder_layer_c<Dtype>());
  shared_ptr< ReorderLayer<Dtype> > layer_chw(create_reorder_layer_chw<Dtype>());
  layer_conly->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer_conly->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  layer_chw->SetUp(this->blob_top_vec_, &(this->blob_result_vec_));
  layer_chw->Forward(this->blob_top_vec_, &(this->blob_result_vec_));

  this->blob_result_->Reshape(128, 256, 2, 3);
  check_reorder_result(*this->blob_bottom_, *this->blob_result_);
}

TYPED_TEST(ReorderLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr< ReorderLayer<Dtype> > layer_conly(create_reorder_layer_c<Dtype>());
  shared_ptr< ReorderLayer<Dtype> > layer_chw(create_reorder_layer_chw<Dtype>());
  vector<bool> propagate_down;
  propagate_down.push_back(true);

  layer_conly->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer_conly->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  layer_chw->SetUp(this->blob_top_vec_, &(this->blob_result_vec_));
  layer_chw->Forward(this->blob_top_vec_, &(this->blob_result_vec_));

  this->blob_result_->Reshape(128, 256, 2, 3);
  check_reorder_result(*this->blob_bottom_, *this->blob_result_);

  caffe_copy(this->blob_result_->count(),
        reinterpret_cast<const Dtype *>(this->blob_result_->cpu_data()),
        reinterpret_cast<Dtype *>(this->blob_result_->mutable_cpu_diff()));
  layer_chw->Backward(this->blob_result_vec_, propagate_down, &(this->blob_top_vec_));
  layer_conly->Backward(this->blob_top_vec_, propagate_down, &(this->blob_bottom_vec_));
  const Dtype *src = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_data());
  const Dtype *dst = reinterpret_cast<const Dtype *>(this->blob_bottom_->cpu_diff());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
      EXPECT_EQ(src[i], dst[i]);
  }
}

// }}}

}  // namespace caffe
