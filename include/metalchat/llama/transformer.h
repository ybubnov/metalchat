#pragma once

#include <optional>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/functional.h>
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

    device& _m_device;

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
      _m_device(device)
    {}

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, const std::optional<Mask> mask, std::size_t start_pos = 0)
    {
        auto norm = _m_attention_norm(input);
        auto h = fn::sum(input, _m_attention(norm, mask, start_pos), _m_device);

        auto ff_norm = _m_ff_norm(h);
        return fn::sum(h, _m_ff(ff_norm), _m_device);
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
