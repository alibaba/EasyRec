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

class LoadEmbedOp: public OpKernel {
 public:
  explicit LoadEmbedOp(OpKernelConstruction* context)
      : OpKernel(context)
  {
     OP_REQUIRES_OK(context, context->GetAttr("task_index", &task_index_));
     OP_REQUIRES_OK(context, context->GetAttr("task_num", &task_num_));
     OP_REQUIRES_OK(context, context->GetAttr("embed_dim", &embed_dim_));
     OP_REQUIRES_OK(context, context->GetAttr("embed_part_size", &embed_part_size_));
     OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
  }

  int get_embed_part_id(const std::string & embed_file_path) const {
    // embed-input_layer__all_fea__embedding_weights:0_part-0.bin
    size_t tmp_pos = embed_file_path.rfind('-', embed_file_path.size() - 5);
    if (tmp_pos == std::string::npos) {
      LOG(ERROR) << "'-' is not found in embed_file_path=" << embed_file_path;
      return -1;
    }
    std::string token = embed_file_path.substr(tmp_pos + 1,
         embed_file_path.size() - 4);
    return std::atoi(token.c_str());
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* file_name_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ckpt_path", &file_name_t));

    tstring file_name = file_name_t->flat<tstring>()(0);
    tstring folder = file_name + "-embedding/";
    tstring prefix = var_name_ + "-part-";

    LOG(INFO) << "task[" << task_index_ << "] file_name=" << file_name
	      << " folder=" << folder << " prefix=" << prefix;

    DIR* pdir = opendir(folder.c_str());
    struct dirent* ent = nullptr;

    std::vector<std::string> embed_files;
    while((ent = readdir(pdir))) {
      if (ent->d_type & DT_REG) {
        std::string name = ent->d_name;
        if (name.find(prefix) == std::string::npos) {
          continue;
        }
        if (name.find(".bin") != std::string::npos) {
          std::string embed_path = folder + name;
          embed_files.push_back(embed_path);
        }
      }
    }
    ::closedir(pdir);

    std::sort(embed_files.begin(), embed_files.end());

    // output shape
    TensorShape val_output_shape({embed_part_size_, embed_dim_});
    Tensor * out_vals_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("vals", val_output_shape, &out_vals_t));

    float * out_val_ptr = (float *)out_vals_t->tensor_data().data();
    const int part_embed_flt_cnt = embed_part_size_ * embed_dim_;
    // memset(out_val_ptr, 0, sizeof(float) * part_embed_flt_cnt);

    const int total_embed_cnt = embed_part_size_ * task_num_;
    const int embed_part_cnt_o = embed_files.size();
    int part_update_cnt = 0;
    for(const auto & embed_file : embed_files) {
      LOG(INFO) << "task[" << task_index_ << "] will load embed_file: " << embed_file;
      std::ifstream fin(embed_file.c_str());
      fin.seekg(0, fin.end);
      const size_t file_len = fin.tellg();
      fin.seekg(0, fin.beg);

      const size_t embed_flt_cnt_o = file_len / sizeof(float);
      std::vector<float> part_embed_o(embed_flt_cnt_o);
      fin.read((char *)(part_embed_o.data()), file_len);
      fin.close();

      const int part_id_o = get_embed_part_id(embed_file);
      const size_t embed_id_cnt_o = embed_flt_cnt_o / embed_dim_;
      for(int embed_id_o = 0; embed_id_o < embed_id_cnt_o; ++embed_id_o) {
        const int part_id_n = embed_id_o *  embed_part_cnt_o + part_id_o;
        if ((part_id_n % task_num_) == task_index_ &&
            part_id_n < total_embed_cnt) {
          const int embed_id_n = part_id_n / task_num_;
          memcpy(out_val_ptr + embed_id_n * embed_dim_,
                 &part_embed_o[embed_id_o * embed_dim_],
                 sizeof(float) * embed_dim_);
          part_update_cnt++;
        }
      }
    }

    LOG(INFO) << "task[" << task_index_ << "] embed_part_size="
	      << embed_part_size_ << " part_update_cnt="
	      << part_update_cnt;
    OP_REQUIRES(ctx, (part_update_cnt == embed_part_size_ ||
                      part_update_cnt + 1 == embed_part_size_),
      errors::InvalidArgument(
          "part_update_cnt or part_update_cnt + 1 should be equal to "
          "embed_part_size_, but are: ", part_update_cnt,
          " and ", embed_part_size_));

    if (part_update_cnt < embed_part_size_) {
      memset(out_val_ptr + (part_embed_flt_cnt - embed_dim_),
             0, sizeof(float) * embed_dim_);
    }
  }

 private:
  int task_index_;
  int task_num_;
  int embed_dim_;
  int embed_part_size_;
  string var_name_;
};

REGISTER_KERNEL_BUILDER(Name("LoadEmbed").Device(DEVICE_CPU), LoadEmbedOp);

REGISTER_OP("LoadEmbed")
    .Attr("task_index: int")
    .Attr("task_num: int")
    .Attr("embed_dim: int")
    .Attr("embed_part_size: int")
    .Attr("var_name: string")
    .Input("ckpt_path: string")
    .Output("vals: float32")
    .SetIsStateful();

} // end namespace tensorflow
