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
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
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
      : blob_bottom_(new Blob<Dtype>(6, 12 * 2 * 3, 1, 1)),
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
  param->set_channels(12);
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
// }}}

// {{{ReorderLayerTest
template <typename TypeParam>
class ReorderLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReorderLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
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
static void check_reorder_result(const Blob<Dtype> &src, const Blob<Dtype> &dst,
        ReorderParameter param) {
#define DEC_GET_VAR(name)   int name = src.name();
    DEC_GET_VAR(num);
    DEC_GET_VAR(channels);
    DEC_GET_VAR(height);
    DEC_GET_VAR(width);
#undef DEC_GET_VAR
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

  this->blob_result_->Reshape(6, 12, 2, 3);
  check_reorder_result(*this->blob_bottom_, *this->blob_result_,
          *create_reorder_layer_c_param().mutable_reorder_param());
}


#if 0

TYPED_TEST(SliceLayerTest, TestSliceAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_slice_dim(0);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_0_));
  const int top_num = this->blob_bottom_->num() / 2;
  ASSERT_EQ(top_num, this->blob_top_0_->num());
  ASSERT_EQ(top_num, this->blob_top_1_->num());
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_0_));
  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n + 3, c, h, w),
                    this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestSliceAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Slice at 2, 8: should produce output blobs with #channels 2, 6, 4.
  const int kSlicePoint0 = 2;
  const int kSlicePoint1 = 8;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint0);
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint1);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_1_));
  ASSERT_EQ(kSlicePoint0, this->blob_top_0_->channels());
  ASSERT_EQ(kSlicePoint1 - kSlicePoint0, this->blob_top_1_->channels());
  ASSERT_EQ(this->blob_bottom_->channels() - kSlicePoint1,
            this->blob_top_2_->channels());
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_1_));
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w),
              this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w),
              this->blob_top_2_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_slice_dim(0);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
    &(this->blob_top_vec_0_));
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  const int kSlicePoint = 4;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
    &(this->blob_top_vec_0_));
}

#endif

}  // namespace caffe
