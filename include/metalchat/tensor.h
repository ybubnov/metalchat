#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <span>
#include <sstream>
#include <type_traits>
#include <utility>

#include <metalchat/device.h>
#include <metalchat/format.h>


namespace metalchat {


template <typename T> struct array_ref {
    using ptr_type = T*;

    virtual ptr_type
    data()
        = 0;

    virtual const T*
    data() const
        = 0;

    virtual ~array_ref() {}
};


template <typename T> struct weak_ref : public array_ref<T> {
private:
    T* m_data = nullptr;

public:
    weak_ref(T* data)
    : m_data(data)
    {}

    weak_ref(const weak_ref& ref)
    : weak_ref(ref.m_data)
    {}

    T*
    data() override
    {
        return m_data;
    }

    const T*
    data() const override
    {
        return m_data;
    }

    ~weak_ref() { m_data = nullptr; }
};

template <typename T> struct owned_ref : public array_ref<T> {
private:
    T* m_data = nullptr;

public:
    owned_ref(T* data)
    : m_data(data)
    {}

    owned_ref(const owned_ref& ref) = delete;

    ~owned_ref()
    {
        delete[] m_data;
        m_data = nullptr;
        std::cout << "owned_ref::~owned_ref()" << std::endl;
    }

    T*
    data() override
    {
        return m_data;
    }

    const T*
    data() const override
    {
        return m_data;
    }
};


template <typename T> struct device_ref : public array_ref<T> {
    using ptr_type = NS::SharedPtr<MTL::Buffer>;

    ptr_type m_buf;

    device_ref(NS::SharedPtr<MTL::Buffer> buf)
    : m_buf(buf)
    {}

    ~device_ref() { std::cout << "device_ref::~device_ref()" << std::endl; }

    T*
    data() override
    {
        return static_cast<T*>(m_buf->contents());
    }

    const T*
    data() const override
    {
        return static_cast<T*>(m_buf->contents());
    }
};


template <typename T> struct value_ref : public array_ref<T> {
private:
    T m_data;

public:
    using ptr_type = T*;

    value_ref(T data)
    : m_data(data)
    {}

    T*
    data() override
    {
        return &m_data;
    }

    const T*
    data() const override
    {
        return &m_data;
    }
};


template <typename T, template <typename U> class Reference>
    requires(std::derived_from<Reference<T>, array_ref<T>>)
struct tensor_traits {
    using data_type = std::unique_ptr<Reference<T>>;
    using size_type = std::unique_ptr<array_ref<std::size_t>>;

    template <typename V, template <typename U> class ref_type>
    static inline std::unique_ptr<ref_type<V>>
    move(V* values)
    {
        return std::move(std::make_unique<ref_type<V>>(values));
    }

    static inline data_type
    weak_move(T* data)
    {
        return move<T, weak_ref>(data);
    }

    static inline size_type
    weak_move(std::size_t* size)
    {
        return move<std::size_t, weak_ref>(size);
    }
};


template <typename T, std::size_t N, template <typename U> class Reference>
    requires(std::derived_from<Reference<T>, array_ref<T>>)
class tensor_base {
public:
    using traits = tensor_traits<T, Reference>;

    tensor_base(T* data, std::size_t* shape, std::size_t* strides)
    : m_data(traits::weak_move(data)),
      m_shape(traits::weak_move(shape)),
      m_strides(traits::weak_move(strides))
    {}

    tensor_base(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : m_data(std::move(data)),
      m_shape(std::move(shape)),
      m_strides(std::move(strides))
    {}

    tensor_base(tensor_base<T, N, Reference>&& t)
    : m_data(std::move(t.m_data)),
      m_shape(std::move(t.m_shape)),
      m_strides(std::move(t.m_strides))
    {}

    inline T*
    data_ptr()
    {
        return m_data->data();
    }

    inline const T*
    data_ptr() const
    {
        return m_data->data();
    }

    inline std::vector<std::size_t>
    shape() const
    {
        return std::vector(m_shape->data(), m_shape->data() + N);
    }

    inline std::size_t
    stride(std::size_t dim) const
    {
        return m_strides->data()[dim];
    }

    inline std::size_t
    size(std::size_t dim) const
    {
        return m_shape->data()[dim];
    }

    std::span<std::size_t, N> const
    sizes() const
    {
        return std::span(m_shape->data(), N);
    }

    std::size_t
    numel() const
    {
        if (N == 0) {
            return 0;
        }

        std::size_t n = 1;
        for (std::size_t i = 0; i < N; i++) {
            n *= m_shape->data()[i];
        }
        return n;
    }

    virtual void
    data_repr(std::ostream& os, int w) const
    {
        os << "[...]";
    }

    inline const traits::data_type&
    storage() const
    {
        return m_data;
    }

protected:
    traits::data_type m_data = nullptr;
    traits::size_type m_shape = nullptr;
    traits::size_type m_strides = nullptr;
};


template <typename T, std::size_t N, template <typename U> class Reference = weak_ref>
struct tensor_format {
    const tensor_base<T, N, Reference>& tensor;
    const int w;

    tensor_format(const tensor_base<T, N, Reference>& tensor_, const int w_ = 0)
    : tensor(tensor_),
      w(w_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, N, Reference>& tf)
    {
        tf.tensor.data_repr(os, tf.w);
        return os;
    }
};


template <typename T, std::size_t N, template <typename U> class Reference>
std::ostream&
operator<<(std::ostream& os, const tensor_base<T, N, Reference>& t)
{
    os << tensor_format<T, N, Reference>(t, 1) << ", shape=(" << t.shape() << ")";
    return os;
}


template <typename T, std::size_t N, template <typename U> class Reference = weak_ref>
    requires(std::derived_from<Reference<T>, array_ref<T>>)
class tensor : public tensor_base<T, N, Reference> {
public:
    using traits = tensor_traits<T, Reference>;

    tensor(T* data, std::size_t* shape, std::size_t* strides)
    : tensor_base<T, N, Reference>(data, shape, strides)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, N, Reference>(std::move(data), std::move(shape), std::move(strides))
    {}

    tensor<T, N - 1>
    at(std::size_t i)
    {
        auto new_data = this->data_ptr() + this->m_strides->data()[0] * i;
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        return tensor(new_data, new_shape, new_strides);
    }

    const tensor<const T, N - 1>
    at(std::size_t i) const
    {
        auto new_data = this->data_ptr() + this->m_strides->data()[0] * i;
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        return tensor<const T, N - 1>(new_data, new_shape, new_strides);
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return at(i);
    }

    template <typename = void>
        requires(N == 2)
    auto
    t()
    {
        auto new_shape = new std::size_t[N]{*(this->m_shape->data() + 1), *this->m_shape->data()};
        auto new_strides
            = new std::size_t[N]{*(this->m_strides->data() + 1), *this->m_strides->data()};
        return tensor<T, 2, weak_ref>(
            std::move(std::make_unique<weak_ref<T>>(this->m_data->data())),
            std::move(std::make_unique<owned_ref<std::size_t>>(new_shape)),
            std::move(std::make_unique<owned_ref<std::size_t>>(new_strides))
        );
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                os << std::endl << std::setw(w) << "";
            }

            os << "..., " << std::endl << std::setw(w) << "";

            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        }
        os << "]";
    }
};


template <typename T, template <typename U> class Reference>
    requires(std::derived_from<Reference<T>, array_ref<T>>)
class tensor<T, 1, Reference> : public tensor_base<T, 1, Reference> {
public:
    using traits = tensor_traits<T, Reference>;

    tensor(T* data, std::size_t* shape, std::size_t* strides)
    : tensor_base<T, 1, Reference>(data, shape, strides)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, 1, Reference>(std::move(data), std::move(shape), std::move(strides))
    {}

    T&
    operator[](std::size_t i)
    {
        return this->data_ptr()[i];
    }

    const T&
    operator[](std::size_t i) const
    {
        return this->data_ptr()[i];
    }

    auto
    t()
    {
        return tensor<T, 1, weak_ref>(
            this->m_data->data(), this->m_shape->data(), this->m_strides->data()
        );
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            os << std::vector<T>(this->data_ptr(), this->data_ptr() + fmt::edgeitems);
            os << ", ..., ";
            os << std::vector<T>(this->data_ptr() + size - fmt::edgeitems, this->data_ptr() + size);
        } else {
            os << std::vector<T>(this->data_ptr(), this->data_ptr() + size);
        }

        os << "]";
    }
};


template <typename T, template <typename U> class Reference>
    requires(std::derived_from<Reference<T>, array_ref<T>>)
class tensor<T, 0, Reference> : public tensor_base<T, 0, Reference> {
public:
    using traits = tensor_traits<T, Reference>;

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, 0, Reference>(std::move(data), std::move(shape), std::move(strides))
    {}

    void
    data_repr(std::ostream& os, int w) const override
    {
        os << *this->data_ptr();
    }
};


template <typename T, std::size_t N>
    requires(N > 0)
auto
empty(std::size_t (&&sizes)[N])
{
    auto shape_ = std::to_array(sizes);
    auto shape = new std::size_t[N];

    std::size_t numel = 1;
    for (auto i = 0; i < N; i++) {
        numel *= shape_[i];
        shape[i] = shape_[i];
    }

    auto data = new T[numel];
    auto strides = new std::size_t[N];

    strides[N - 1] = 1;
    for (auto i = N - 2; i < N; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    using tensor_type = tensor<T, N, owned_ref>;

    return tensor_type(
        std::move(std::make_unique<owned_ref<T>>(data)),
        std::move(std::make_unique<owned_ref<std::size_t>>(shape)),
        std::move(std::make_unique<owned_ref<std::size_t>>(strides))
    );
}


template <typename T>
auto
scalar(const T& value)
{
    using tensor_type = tensor<T, 0, value_ref>;

    return tensor_type(
        std::move(std::make_unique<value_ref<T>>(value)),
        std::move(std::make_unique<value_ref<std::size_t>>(0)),
        std::move(std::make_unique<value_ref<std::size_t>>(0))
    );
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
empty(std::size_t (&&sizes)[N], device& device)
{
    auto shape_ = std::to_array(sizes);
    auto shape = new std::size_t[N];

    std::size_t numel = 1;
    for (auto i = 0; i < N; i++) {
        numel *= shape_[i];
        shape[i] = shape_[i];
    }

    auto data
        = NS::TransferPtr(device->newBuffer(numel * sizeof(T), MTL::ResourceStorageModeShared));

    auto strides = new std::size_t[N];

    strides[N - 1] = 1;
    for (auto i = N - 2; i < N; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    using tensor_type = tensor<T, N, device_ref>;

    return tensor_type(
        std::move(std::make_unique<device_ref<T>>(data)),
        std::move(std::make_unique<owned_ref<std::size_t>>(shape)),
        std::move(std::make_unique<owned_ref<std::size_t>>(strides))
    );
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value)
{
    auto t = empty<T>(std::move(sizes));
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value, device& device)
{
    auto t = empty<T>(std::move(sizes), device);
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
zeros(std::size_t (&&sizes)[N])
{
    return full<T>(std::move(sizes), 0);
}


using int32_tensor1d = tensor<int32_t, 1>;
using int32_tensor2d = tensor<int32_t, 2>;


} //  namespace metalchat
