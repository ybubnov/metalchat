#include <catch2/catch_test_macros.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor empty", "[tensor::tensor]")
{
    auto t = tensor<float, 4>();

    // Ensure tensor formatting works without exceptions.
    std::cout << t << std::endl;

    REQUIRE(t.dim() == 4);
    REQUIRE(t.numel() == 0);
    REQUIRE(t.container_offset() == 0);
    REQUIRE(t.data_ptr() == nullptr);

    for (std::size_t i = 0; i < t.dim(); i++) {
        REQUIRE(t.size(i) == 0);
        REQUIRE(t.stride(i) == 0);
        REQUIRE(t.offset(i) == 0);
    }

    // Empty tensor should output nothing, use iterator to avoid skipping the
    // cycle by the compiler.
    for (auto it = t.begin(); it != t.end(); ++it) {
        REQUIRE(false);
        std::cout << *it << std::endl;
    }

    auto tt = t.transpose({0, 2, 1, 3});
    for (std::size_t i = 0; i < t.dim(); i++) {
        REQUIRE(t.size(i) == 0);
        REQUIRE(t.stride(i) == 0);
        REQUIRE(t.offset(i) == 0);
    }
}


TEST_CASE("Tensor full", "[tensor::tensor]")
{
    auto t = full<float>({2, 3, 4}, 4.0);
    REQUIRE(t.dim() == 3);
    REQUIRE(t.size(0) == 2);
    REQUIRE(t.size(1) == 3);
    REQUIRE(t.size(2) == 4);
    REQUIRE(t.stride(0) == 12);
    REQUIRE(t.stride(1) == 4);
    REQUIRE(t.stride(2) == 1);
    REQUIRE(t.offset(0) == 0);
    REQUIRE(t.offset(1) == 0);
    REQUIRE(t.offset(2) == 0);
    REQUIRE(t.numel() == 24);

    auto container = t.container();
    for (std::size_t i = 0; i < t.numel(); i++) {
        REQUIRE(container.data()[i] == 4.0);
        container.data()[i] = i;
    }

    std::cout << t << std::endl;
}


TEST_CASE("Tensor at", "[tensor::at]")
{
    auto t = full<float>({2, 3, 4}, 5.0);
    auto u = t.at(1);

    REQUIRE(u.dim() == 2);
    REQUIRE(u.size(0) == 3);
    REQUIRE(u.size(1) == 4);
    REQUIRE(u.stride(0) == 4);
    REQUIRE(u.stride(1) == 1);
    REQUIRE(u.offset(0) == 0);
    REQUIRE(u.offset(1) == 0);

    for (std::size_t i = 0; i < 3; i++) {
        for (std::size_t j = 0; j < 4; j++) {
            REQUIRE((u[i, j]) == 5.0);
            u[i, j] = 2.0;
            REQUIRE((u[i, j]) == 2.0);
        }
    }
}


TEST_CASE("Tensor move assignment", "[tensor::operator=(tensor&&)]")
{
    auto t = rand<float>({3, 2});
    REQUIRE(t.numel() == 6);

    t = rand<float>({4, 2});

    REQUIRE(t.dim() == 2);
    REQUIRE(t.size(0) == 4);
    REQUIRE(t.size(1) == 2);
    REQUIRE(t.numel() == 8);
}


TEST_CASE("Tensor transpose", "[tensor::transpose]")
{
    auto x = rand<float>({2, 3, 4});
    REQUIRE(x.size(0) == 2);
    REQUIRE(x.size(1) == 3);
    REQUIRE(x.size(2) == 4);

    auto x_t = x.transpose({0, 2, 1});

    REQUIRE(x.dim() == 3);
    REQUIRE(x_t.size(0) == 2);
    REQUIRE(x_t.size(1) == 4);
    REQUIRE(x_t.size(2) == 3);

    for (std::size_t i = 0; i < x.size(0); i++) {
        for (std::size_t j = 0; j < x.size(1); j++) {
            for (std::size_t k = 0; k < x.size(2); k++) {
                REQUIRE((x[i, j, k]) == (x_t[i, k, j]));
            }
        }
    }
}


TEST_CASE("Tensor slice transpose", "[tensor::transpose]")
{
    auto x = rand<float>({5, 4, 3, 2});
    auto y = x[slice(0, 1), slice(1, 3), slice(0, 2), slice(1, 2)];
    REQUIRE(y.size(0) == 1);
    REQUIRE(y.size(1) == 2);
    REQUIRE(y.size(2) == 2);
    REQUIRE(y.size(3) == 1);

    auto y_t = y.transpose({1, 0, 3, 2});
    REQUIRE(y_t.size(0) == 2);
    REQUIRE(y_t.size(1) == 1);
    REQUIRE(y_t.size(2) == 1);
    REQUIRE(y_t.size(3) == 2);

    std::fill(y_t.begin(), y_t.end(), 0.0);

    for (std::size_t i = 0; i < 1; i++) {
        for (std::size_t j = 1; j < 3; j++) {
            for (std::size_t k = 0; k < 2; k++) {
                for (std::size_t l = 1; l < 2; l++) {
                    REQUIRE(x[i][j][k][l] == 0.0);
                }
            }
        }
    }
}


TEST_CASE("Tensor transpose in scope", "[tensor::transpose]")
{
    auto x = []() {
        auto x = full<float>({3, 4, 2, 2}, 7.0);
        return x.transpose({0, 2, 3, 1});
    }();

    REQUIRE(x.dim() == 4);
    REQUIRE(x.size(0) == 3);
    REQUIRE(x.size(1) == 2);
    REQUIRE(x.size(2) == 2);
    REQUIRE(x.size(3) == 4);

    for (const auto& v : x) {
        REQUIRE(v == 7.0);
    }
}


TEST_CASE("Tensor format", "[tensor::ostream]")
{
    auto t0 = scalar<float>(5.0);
    auto t1 = full<float>({3}, 6.0);
    auto t2 = full<float>({3, 4}, 7.0);
    auto t3 = full<float>({3, 4, 5}, 8.0);

    std::cout << t0 << std::endl;
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << t3 << std::endl;
}


TEST_CASE("Tensor view", "[tensor::view]")
{
    auto t = rand<float>({3, 4, 2});

    auto t0 = t.view({24});
    REQUIRE(t0.dim() == 1);
    REQUIRE(t0.size(0) == 24);

    t0[23] = 15.0;
    REQUIRE(t[2][3][1] == 15.0);
}


TEST_CASE("Tensor reshape unsqueeze", "[tensor::view]")
{
    auto t = rand<float>({4, 5, 2});
    auto t0 = t.view({4, 5, 2, -1});

    REQUIRE(t0.dim() == 4);
    REQUIRE(t0.size(0) == 4);
    REQUIRE(t0.size(1) == 5);
    REQUIRE(t0.size(2) == 2);
    REQUIRE(t0.size(3) == 1);
    REQUIRE(t0.stride(0) == 10);
    REQUIRE(t0.stride(1) == 2);
    REQUIRE(t0.stride(2) == 1);
    REQUIRE(t0.stride(3) == 1);

    REQUIRE(t.numel() == t0.numel());
    t0[3][4][1][0] = 100.0;
    REQUIRE(t[3][4][1] == 100.0);
}


TEST_CASE("Tensor expand dimensions", "[tensor::expand_dims]")
{
    auto t = rand<float>({6, 3, 8, 2});
    auto t0 = t.expand_dims(2);

    REQUIRE(t0.dim() == 5);
    REQUIRE(t0.size(0) == 6);
    REQUIRE(t0.size(1) == 3);
    REQUIRE(t0.size(2) == 1);
    REQUIRE(t0.size(3) == 8);
    REQUIRE(t0.size(4) == 2);
    REQUIRE(t0.numel() == t.numel());
}


TEST_CASE("Tensor flatten dimensions", "[tensor::flatten]")
{
    auto t = rand<float>({2, 4, 8, 10});
    auto t0 = t.flatten<2>();

    REQUIRE(t0.dim() == 2);
    REQUIRE(t0.size(0) == 64);
    REQUIRE(t0.size(1) == 10);
}


TEST_CASE("Tensor initializer list", "[tensor::initializer_list]")
{
    auto t1 = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    REQUIRE(t1.dim() == 1);
    REQUIRE(t1.size(0) == 5);

    for (std::size_t i = 0; i < t1.size(0); i++) {
        REQUIRE((t1[i]) == float(i + 1.0f));
    }

    auto t2 = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f}});
    REQUIRE(t2.dim() == 2);
    REQUIRE(t2.size(0) == 2);
    REQUIRE(t2.size(1) == 3);

    REQUIRE((t2[0, 0]) == 1.0f);
    REQUIRE((t2[0, 1]) == 2.0f);
    REQUIRE((t2[0, 2]) == 3.0f);
    REQUIRE((t2[1, 0]) == 4.0f);
    REQUIRE((t2[1, 1]) == 5.0f);
    REQUIRE((t2[1, 2]) == 0.0f);
}
