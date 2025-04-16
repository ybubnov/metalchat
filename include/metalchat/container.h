#pragma once


#include <metalchat/device.h>


namespace metalchat {


template <typename T> struct array_ref {
    using ptr_type = T*;
    using value_type = T;

    virtual ptr_type
    data()
        = 0;

    virtual const T*
    data() const
        = 0;

    T*
    operator*()
    {
        return data();
    }

    virtual ~array_ref() {}
};


template <typename T>
concept ContiguousContainer = requires { typename T::value_type; }
                              && std::derived_from<T, array_ref<typename T::value_type>>;


template <ContiguousContainer Container, class OutputIt>
OutputIt
copy(const Container& container, std::size_t count, OutputIt d_first)
{
    auto first = container.data();
    auto last = container.data() + count;
    for (; first != last; ++first, ++d_first)
        *d_first = *first;
    return d_first;
}


template <ContiguousContainer Container, class OutputIt>
OutputIt
reverse_copy(const Container& container, std::size_t count, OutputIt d_first)
{
    auto first = container.data();
    auto last = first + count;
    for (; first != last; ++d_first)
        *d_first = *(--last);
    return d_first;
}


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


template <typename T>
auto
make_weak(T* data)
{
    return std::make_unique<weak_ref<T>>(data);
}


template <typename T> struct owning_ref : public array_ref<T> {
private:
    T* m_data = nullptr;

public:
    owning_ref(T* data)
    : m_data(data)
    {}

    owning_ref(const owning_ref& ref) = delete;

    ~owning_ref()
    {
        delete[] m_data;
        m_data = nullptr;
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


template <typename T>
auto
make_owning(T* data)
{
    return std::make_unique<owning_ref<T>>(data);
}


template <typename T> struct device_ref : public array_ref<T> {
    using ptr_type = NS::SharedPtr<MTL::Buffer>;

    ptr_type m_buf;

    device_ref(NS::SharedPtr<MTL::Buffer> buf)
    : m_buf(buf)
    {}

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

    ptr_type
    storage()
    {
        return m_buf;
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


template <typename T>
auto
make_value(T data)
{
    return std::make_unique<value_ref<T>>(data);
}


} // namespace metalchat
