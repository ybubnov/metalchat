#pragma once

#include <functional>
#include <vector>

#include <metalchat/metal.h>


namespace metalchat {


template <typename T> struct memory_container {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

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
concept contiguous_container = requires {
    typename Container::value_type;
} && std::derived_from<Container, memory_container<typename Container::value_type>>;


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
    data()
    {
        return _m_data;
    }

    inline const_pointer
    data() const
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
    data()
    {
        return _m_data;
    }

    inline const_pointer
    data() const
    {
        return _m_data;
    }

    template <typename U> requires std::convertible_to<U, T>
    operator random_memory_container<U>() const
    {
        return random_memory_container<U>(_m_data);
    }
};


template <typename T> struct vector_memory_container : public memory_container<T> {
private:
    std::vector<T> _m_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    vector_memory_container(std::vector<T>&& data)
    : _m_data(std::move(data))
    {}

    vector_memory_container()
    : _m_data()
    {}

    inline pointer
    data()
    {
        return _m_data.data();
    }

    inline const value_type*
    data() const
    {
        const value_type* ptr = _m_data.data();
        return ptr;
    }
};


template <typename T> struct hardware_memory_container : public memory_container<T> {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    metal::shared_buffer _m_mem;

    hardware_memory_container(metal::shared_buffer mem)
    : _m_mem(mem)
    {}

    pointer
    data()
    {
        return static_cast<T*>(metal::data(_m_mem));
    }

    const_pointer
    data() const
    {
        return static_cast<const_pointer>(metal::data(_m_mem));
    }

    template <typename U> requires std::convertible_to<U, T>
    operator hardware_memory_container<U>() const
    {
        return hardware_memory_container<U>(_m_mem);
    }

    metal::shared_buffer
    storage() const
    {
        return _m_mem;
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
    data()
    {
        return &_m_data;
    }

    const_pointer
    data() const
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
