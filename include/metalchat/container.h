#pragma once

#include <metalchat/metal.h>


namespace metalchat {


template <typename T> struct memory_container {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    virtual pointer
    data()
        = 0;

    virtual const_pointer
    data() const
        = 0;

    pointer
    operator*()
    {
        return data();
    }

    virtual ~memory_container() {}
};


template <typename Container>
concept contiguous_container = requires(Container c) {
    typename Container::value_type;

    std::derived_from<Container, memory_container<typename Container::value_type>>;
};


template <typename T> struct reference_memory_container : public memory_container<T> {
private:
    T* _m_data = nullptr;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    reference_memory_container(pointer data)
    : _m_data(data)
    {}

    reference_memory_container(const reference_memory_container& ref)
    : reference_memory_container(ref._m_data)
    {}

    inline pointer
    data() override
    {
        return _m_data;
    }

    inline const_pointer
    data() const override
    {
        return _m_data;
    }

    ~reference_memory_container() { _m_data = nullptr; }
};


template <typename T>
auto
make_weak(T* data)
{
    return std::make_shared<reference_memory_container<T>>(data);
}


template <typename T> struct random_memory_container : public memory_container<T> {
private:
    T* _m_data = nullptr;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    random_memory_container(T* data)
    : _m_data(data)
    {}

    random_memory_container(const random_memory_container& ref) = delete;

    ~random_memory_container()
    {
        delete[] _m_data;
        _m_data = nullptr;
    }

    inline pointer
    data() override
    {
        return _m_data;
    }

    inline const_pointer
    data() const override
    {
        return _m_data;
    }
};


template <typename T> struct hardware_memory_container : public memory_container<T> {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    NS::SharedPtr<MTL::Buffer> m_buf;

    hardware_memory_container(NS::SharedPtr<MTL::Buffer> buf)
    : m_buf(buf)
    {}

    ~hardware_memory_container() { m_buf.reset(); }

    pointer
    data() override
    {
        return static_cast<T*>(m_buf->contents());
    }

    const_pointer
    data() const override
    {
        return static_cast<T*>(m_buf->contents());
    }

    NS::SharedPtr<MTL::Buffer>
    storage()
    {
        return m_buf;
    }
};


template <typename T> struct scalar_memory_container : public memory_container<T> {
private:
    T _m_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    scalar_memory_container(T data)
    : _m_data(data)
    {}

    pointer
    data() override
    {
        return &_m_data;
    }

    const_pointer
    data() const override
    {
        return const_pointer(&_m_data);
    }
};


template <typename T>
auto
make_value(T data)
{
    return std::make_shared<scalar_memory_container<T>>(data);
}


} // namespace metalchat
