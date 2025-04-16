#pragma once


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
reverse_copy(const Container& first, std::size_t count, OutputIt d_first)
{
    auto last = first.data() + count;
    for (; first.data() != last; ++d_first)
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


} // namespace metalchat
