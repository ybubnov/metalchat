#pragma once

#include <optional>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/llama/attention.h>
#include <metalchat/llama/feed_forward.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace llama {

template <typename T, ContiguousContainer Container> class transformer {
private:
    attention<T, device_ref<T>> _m_attention;
    nn::rmsnorm<T, Container> _m_attention_norm;

    feed_forward<T, device_ref<T>> _m_ff;
    nn::rmsnorm<T, Container> _m_ff_norm;

    sum<T> _m_sum;

public:
    transformer(transformer&&) = default;
    transformer(const transformer&) = delete;

    transformer(
        attention<T, device_ref<T>>&& attention,
        nn::rmsnorm<T, Container>&& attention_norm,
        feed_forward<T, device_ref<T>>&& ff,
        nn::rmsnorm<T, Container>&& ff_norm,
        device& device
    )
    : _m_attention(std::move(attention)),
      _m_attention_norm(std::move(attention_norm)),
      _m_ff(std::move(ff)),
      _m_ff_norm(std::move(ff_norm)),
      _m_sum(device)
    {}

    template <immutable_tensor3d InputTensor, immutable_tensor2d MaskTensor>
    auto
    operator()(InputTensor input, const std::optional<MaskTensor> mask, std::size_t start_pos = 0)
    {
        auto norm = _m_attention_norm(input);
        auto h = _m_sum(input, _m_attention(norm, mask, start_pos));

        auto ff_norm = _m_ff_norm(h);
        return _m_sum(h, _m_ff(ff_norm));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const transformer& t)
    {
        os << "llama::transformer<" << type_traits<T>::name() << ">()";
        return os;
    }
};


} // namespace llama
} // namespace metalchat
