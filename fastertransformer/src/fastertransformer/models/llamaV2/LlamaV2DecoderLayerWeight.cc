/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llamaV2/LlamaV2DecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
LlamaV2DecoderLayerWeight<T>::LlamaV2DecoderLayerWeight(const int  head_num,
                                                        const int  kv_head_num,
                                                        const int  size_per_head,
                                                        const int  inter_size,
                                                        const int  tensor_para_size,
                                                        const int  tensor_para_rank,
                                                        const bool use_gptj_residual,
                                                        const int  int8_mode):
    head_num_(head_num),
    kv_head_num_(kv_head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    use_gptj_residual_(use_gptj_residual)
{
    mallocWeights();
    setWeightPtr();

    FT_CHECK_WITH_INFO(int8_mode_ != 2, "LlamaV2 doesn't support int8_model == 2");
    FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value && int8_mode_ == 1),
                       "Weight only quant does not work with FP32 compute.");
}

template<typename T>
LlamaV2DecoderLayerWeight<T>::LlamaV2DecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
LlamaV2DecoderLayerWeight<T>::~LlamaV2DecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 14; i++) {
            if (!use_gptj_residual_ && i != attention_dense_bias_weight_id) {
                cudaFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        post_attention_layernorm_weights.beta                 = nullptr;
        post_attention_layernorm_weights.gamma                = nullptr;

        ffn_weights.intermediate_weight.kernel  = nullptr;
        ffn_weights.intermediate_weight.bias    = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.intermediate_weight2.bias   = nullptr;
        ffn_weights.output_weight.kernel        = nullptr;
        ffn_weights.output_weight.bias          = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr.size(); i++) {
                if (int8_weights_ptr[i] != nullptr) {
                    deviceFree(int8_weights_ptr[i]);
                }
            }

            if (int8_mode_ == 1) {
                for (int i = 0; i < weight_only_scale_ptr.size(); i++) {
                    if (weight_only_scale_ptr[i] != nullptr) {
                        deviceFree(weight_only_scale_ptr[i]);
                    }
                }
            }

            self_attention_weights.query_weight.int8_kernel                        = nullptr;
            self_attention_weights.query_weight.weight_only_quant_scale            = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel             = nullptr;
            self_attention_weights.attention_output_weight.weight_only_quant_scale = nullptr;

            // intermediate_weight => gate_proj;  intermediate_weight2 => up_proj; output_weight =>down_proj.
            ffn_weights.intermediate_weight.int8_kernel              = nullptr;
            ffn_weights.intermediate_weight.weight_only_quant_scale  = nullptr;
            ffn_weights.intermediate_weight2.int8_kernel             = nullptr;
            ffn_weights.intermediate_weight2.weight_only_quant_scale = nullptr;
            ffn_weights.output_weight.int8_kernel                    = nullptr;
            ffn_weights.output_weight.weight_only_quant_scale        = nullptr;
        }

        is_maintain_buffer = false;
    }
}

template<typename T>
void LlamaV2DecoderLayerWeight<T>::copyFrom(const LlamaV2DecoderLayerWeight& other)
{
    int qkv_size = hidden_units_ + 2 * size_per_head_ * kv_head_num_;
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    if (!use_gptj_residual_) {
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    }
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], qkv_size * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else {
        cudaD2Dcpy(int8_weights_ptr[0], other.int8_weights_ptr[0], qkv_size * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[4], other.int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);

        if (int8_mode_ == 1) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], qkv_size / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);

            // TODO: 不太清楚这里存的缩放因子对应的是gate_pro_weight 还是给
            // up_proj/down_proj用的，后面做一下验证，回来再改一下
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], inter_size_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[4], other.weight_only_scale_ptr[4], hidden_units_);
        }
    }
}

template<typename T>
LlamaV2DecoderLayerWeight<T>::LlamaV2DecoderLayerWeight(const LlamaV2DecoderLayerWeight& other):
    head_num_(other.head_num_),
    kv_head_num_(other.kv_head_num_),
    size_per_head_(other.size_per_head_),
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    use_gptj_residual_(other.use_gptj_residual_)
{
    mallocWeights();
    copyFrom(other);
    setWeightPtr();
}

template<typename T>
LlamaV2DecoderLayerWeight<T>& LlamaV2DecoderLayerWeight<T>::operator=(const LlamaV2DecoderLayerWeight& other)
{
    head_num_          = other.head_num_;
    kv_head_num_       = other.kv_head_num_;
    size_per_head_     = other.size_per_head_;
    hidden_units_      = other.hidden_units_;
    inter_size_        = other.inter_size_;
    tensor_para_size_  = other.tensor_para_size_;
    tensor_para_rank_  = other.tensor_para_rank_;
    int8_mode_         = other.int8_mode_;
    use_gptj_residual_ = other.use_gptj_residual_;

    mallocWeights();

    copyFrom(other);
    setWeightPtr();
    return *this;
}

template<typename T>
void LlamaV2DecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    const std::string rank_spec = std::to_string(tensor_para_rank_);

    // fill all bias to zeros
    deviceFill(weights_ptr[0], (size_t)hidden_units_, (T)0.0);
    loadWeightFromBin<T>(
        weights_ptr[1], {(size_t)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);

    int qkv_size = hidden_units_ + 2 * size_per_head_ * kv_head_num_;

    deviceFill(weights_ptr[3], (size_t)(3 * hidden_units_ / tensor_para_size_), (T)0.0);

    if (!use_gptj_residual_) {
        deviceFill(weights_ptr[5], (size_t)hidden_units_, (T)0.0);
    }

    // FIXME(sunpeng17): check if the weights are correct
    // loadWeightFromBin<T>(weights_ptr[6],
    //                      {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
    //                      dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
    //                      model_file_type);

    deviceFill(weights_ptr[7], (size_t)(inter_size_ / tensor_para_size_), (T)0.0);

    deviceFill(weights_ptr[9], (size_t)(inter_size_ / tensor_para_size_), (T)0.0);

    // loadWeightFromBin<T>(weights_ptr[10],
    //                      {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
    //                      dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
    //                      model_file_type);

    deviceFill(weights_ptr[11], (size_t)(hidden_units_), (T)0.0);

    deviceFill(weights_ptr[12], (size_t)(hidden_units_), (T)0.0);
    loadWeightFromBin<T>(
        weights_ptr[13], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

    if (int8_mode_ == 0) {
        loadWeightFromBin<T>(weights_ptr[2],
                             {(size_t)hidden_units_, (size_t)(qkv_size / tensor_para_size_)},
                             dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[4],
                             {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                             dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[6],
                             {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                             dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[8],
                             {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                             dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[10],
                             {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                             dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                             model_file_type);
    }
    else if (int8_mode_ == 1) {
        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[0],
                                                     weight_only_scale_ptr[0],
                                                     {(size_t)hidden_units_, (size_t)(qkv_size / tensor_para_size_)},
                                                     dir_path + ".attention.query_key_value.weight." + rank_spec
                                                         + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(
            int8_weights_ptr[1],
            weight_only_scale_ptr[1],
            {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
            dir_path + ".attention.dense.weight." + rank_spec + ".bin",
            model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[2],
                                                     weight_only_scale_ptr[2],
                                                     {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                                                     dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[3],
                                                     weight_only_scale_ptr[3],
                                                     {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                                                     dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                                                     model_file_type);
        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[4],
                                                     weight_only_scale_ptr[4],
                                                     {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                                                     dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                                                     model_file_type);
    }
}

template<typename T>
void LlamaV2DecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = use_gptj_residual_ ? nullptr : weights_ptr[5];

    ffn_weights.intermediate_weight.kernel  = weights_ptr[6];
    ffn_weights.intermediate_weight.bias    = weights_ptr[7];
    ffn_weights.intermediate_weight2.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight2.bias   = weights_ptr[9];
    ffn_weights.output_weight.kernel        = weights_ptr[10];
    ffn_weights.output_weight.bias          = weights_ptr[11];

    post_attention_layernorm_weights.beta  = weights_ptr[12];
    post_attention_layernorm_weights.gamma = weights_ptr[13];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel            = int8_weights_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel = int8_weights_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel                = int8_weights_ptr[2];
        ffn_weights.intermediate_weight2.int8_kernel               = int8_weights_ptr[3];
        ffn_weights.output_weight.int8_kernel                      = int8_weights_ptr[4];

        if (int8_mode_ == 1) {
            self_attention_weights.query_weight.weight_only_quant_scale            = weight_only_scale_ptr[0];
            self_attention_weights.attention_output_weight.weight_only_quant_scale = weight_only_scale_ptr[1];
            ffn_weights.intermediate_weight.weight_only_quant_scale                = weight_only_scale_ptr[2];
            ffn_weights.intermediate_weight2.weight_only_quant_scale               = weight_only_scale_ptr[3];
            ffn_weights.output_weight.weight_only_quant_scale                      = weight_only_scale_ptr[4];
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void LlamaV2DecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);  // pre layernorm beta
    deviceMalloc(&weights_ptr[1], hidden_units_);  // pre layernorm gamma
    int qkv_size = hidden_units_ + 2 * size_per_head_ * kv_head_num_;
    // deviceMalloc(&weights_ptr[2], hidden_units_ * qkv_size / tensor_para_size_); // qkv kernel
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);  // qkv bias
    // deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_); // attention output weight
    if (!use_gptj_residual_) {
        deviceMalloc(&weights_ptr[5], hidden_units_);  // attention output bias
    }

    // deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_); // intermediate_weight kernel
    deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);  // intermediate_weight bias
    // deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_); // intermediate_weight2 kernel
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);  // intermediate_weight2 bias
    // deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_); // output_weight kernel
    deviceMalloc(&weights_ptr[11], hidden_units_);  // output_weight bias
    deviceMalloc(&weights_ptr[12], hidden_units_);  // post attn layernorm beta
    deviceMalloc(&weights_ptr[13], hidden_units_);  // post attn layernorm gamma

    if (int8_mode_ == 0) {
        deviceMalloc(&weights_ptr[2], qkv_size * hidden_units_ / tensor_para_size_);       // qkv weight
        deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);  // attention output weight
        deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);    // intermediate_weight kernel
        deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);   // intermediate_weight2 kernel
        deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);  // output_weight kernel
    }
    else {
        // Alloc FFN and Attention int8 weights
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * qkv_size / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);

        if (int8_mode_ == 1) {
            // Alloc scales for weight only quant for attention and FFN weights
            deviceMalloc(&weight_only_scale_ptr[0], qkv_size / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[1], hidden_units_);
            deviceMalloc(&weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[3], inter_size_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[4], hidden_units_);
        }
    }
}

template struct LlamaV2DecoderLayerWeight<float>;
template struct LlamaV2DecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template class LlamaV2DecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
