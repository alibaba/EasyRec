#include <dirent.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <exception>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/framework/resource_handle.h"

namespace tensorflow {


class LoadKVEmbedOp: public OpKernel {
 public:
  explicit LoadKVEmbedOp(OpKernelConstruction* context)
      : OpKernel(context)
  {
     OP_REQUIRES_OK(context, context->GetAttr("task_index", &task_index_));
     OP_REQUIRES_OK(context, context->GetAttr("task_num", &task_num_));
     OP_REQUIRES_OK(context, context->GetAttr("embed_dim", &embed_dim_));
     OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* file_name_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ckpt_path", &file_name_t));

    tstring file_name = file_name_t->flat<tstring>()(0);

    tstring folder = file_name + "-embedding/";

    tstring prefix = var_name_ + "-part-";

    LOG(INFO) << "file_name=" << file_name << " folder=" << folder << " prefix=" << prefix;

    DIR* pdir = opendir(folder.c_str());
    struct dirent* ent = nullptr;

    std::vector<int64_t *> key_ptr_vec;
    std::vector<float *> val_ptr_vec;
    std::vector<int> key_num_vec;
    int all_worker_total_keys = 0;
    while((ent = readdir(pdir))) {
      if (ent->d_type & DT_REG) {
        std::string name = ent->d_name;
        if (name.find(prefix) == std::string::npos) {
          continue;
        }
        if (name.find(".key") != std::string::npos) {
          std::string key_path = folder + name;
          LOG(INFO) << "load keys from " << key_path;
          std::ifstream fin(key_path.c_str(), std::ifstream::binary);
          fin.seekg(0, fin.end);
          size_t file_len = fin.tellg();
          fin.seekg(0, fin.beg);
          const size_t key_num = file_len / sizeof(int64_t);
          key_num_vec.push_back(key_num);
          int64_t * key_buf = new int64_t[key_num];
          fin.read((char *)key_buf, file_len);
          fin.close();
          key_ptr_vec.push_back(key_buf);

          LOG(INFO) << "load keys from " << key_path << " key_num=" << key_num;

          std::string val_path = key_path.substr(0, key_path.size()-4) + ".val";
          LOG(INFO) << "load vals from " << val_path;
          fin.open(val_path.c_str(), std::ifstream::binary);
          if (! fin) {
            char err_msg_buf[1024];
            snprintf(err_msg_buf, 1024, "error: file does not exists: %s",
              val_path.c_str());
            LOG(ERROR) << err_msg_buf;
            throw std::runtime_error(err_msg_buf);
          }
          fin.seekg(0, fin.end);
          file_len = fin.tellg();
          if (file_len != key_num * embed_dim_ * sizeof(float)) {
            fin.close();
            char err_msg_buf[1024];
            snprintf(err_msg_buf, 1024,
                "error: key_num[%ld] does not match with val_num[%ld], embed_dim=[%d]",
                key_num, file_len / sizeof(float), embed_dim_);
            LOG(ERROR) << err_msg_buf;
            throw std::runtime_error(err_msg_buf);
          }
          fin.seekg(0, fin.beg);
          float * val_buf = new float[key_num * embed_dim_];
          fin.read((char *)val_buf, file_len);
          fin.close();
          val_ptr_vec.push_back(val_buf);

          all_worker_total_keys += key_num;
          LOG(INFO) << "all_worker_total_keys=" << all_worker_total_keys;
        }
      }
    }
    closedir(pdir);

    // filter key by index
    const int vec_num = key_num_vec.size();
    std::vector<std::pair<int, int> > sel_ids;
    sel_ids.reserve(all_worker_total_keys / task_num_);
    int total_keys = 0;
    for(int i = 0; i < key_ptr_vec.size(); ++i) {
      const int64_t * key_ptr = key_ptr_vec[i];
      const int key_num = key_num_vec[i];
      for(int j = 0; j < key_num; ++j) {
	int assign_id = key_ptr[j] % task_num_;
	if (assign_id < 0) {
          assign_id += task_num_;
	}
        if (assign_id == task_index_) {
          total_keys++;
          sel_ids.push_back(std::pair<int, int>(i,j));
        }
      }
    }

    LOG(INFO) << "task[" << task_index_  << "/" << task_num_
        << "] all_worker_total_keys=" << all_worker_total_keys
        << " load_part_num=" << vec_num
        << " total_keys=" << total_keys << " embed_dim=" << embed_dim_;

    // output shape
    TensorShape key_output_shape({total_keys});
    Tensor * out_keys_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("keys", key_output_shape, &out_keys_t));
    TensorShape val_output_shape({total_keys, embed_dim_});
    Tensor * out_vals_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("vals", val_output_shape, &out_vals_t));

    {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(sel_ids.begin(), sel_ids.end(), g);
    }

    int64_t * key_ptr = (int64_t*)out_keys_t->tensor_data().data();
    float * val_ptr = (float*)out_vals_t->tensor_data().data();
    for(auto iter = sel_ids.begin(); iter != sel_ids.end(); ++iter) {
      const int64_t * src_key_ptr = key_ptr_vec[iter->first] + iter->second;
      const float * src_val_ptr = val_ptr_vec[iter->first] + iter->second * embed_dim_;
      key_ptr[0] = src_key_ptr[0];
      memcpy(val_ptr, src_val_ptr, sizeof(float) * embed_dim_);
      key_ptr += 1;
      val_ptr += embed_dim_;
    }

    for(int i = 0; i < vec_num; ++i) {
      delete [] key_ptr_vec[i];
      delete [] val_ptr_vec[i];
    }
  }

 private:
  int task_index_;
  int task_num_;
  int embed_dim_;
  string var_name_;
};

REGISTER_KERNEL_BUILDER(Name("LoadKVEmbed").Device(DEVICE_CPU), LoadKVEmbedOp);

REGISTER_OP("LoadKVEmbed")
    .Attr("task_index: int")
    .Attr("task_num: int")
    .Attr("embed_dim: int")
    .Attr("var_name: string")
    .Input("ckpt_path: string")
    .Output("keys: int64")
    .Output("vals: float32")
    .SetIsStateful();

} // end namespace tensorflow
