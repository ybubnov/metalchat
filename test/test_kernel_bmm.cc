#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Matmul simple", "[kernel::bmm]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::bmm<bf16> mm(gpu0);

    auto input1 = shared_tensor(full<bf16>({32, 32}, 2.0));
    auto input2 = shared_tensor(full<bf16>({32, 32}, 3.0));
    auto output = mm(input1, input2).get();

    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 32);
    REQUIRE(output.size(1) == 32);
}


TEST_CASE("Matmul single batch multiplication", "[kernel::bmm]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::bmm<float> mm(gpu0);

    auto input1 = shared_tensor(rand<float>({1, 5, 2048}));     // b, i, j
    auto input2 = shared_tensor(rand<float>({8192, 2048}).t()); // j, k

    auto output = mm(input1, input2).get();
    std::cout << output << std::endl;

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 5);
    REQUIRE(output.size(2) == 8192);

    for (std::size_t batch = 0; batch < input1.size(0); batch++) {
        for (std::size_t i = 0; i < input1.size(1); i++) {
            for (std::size_t k = 0; k < input2.size(1); k++) {
                float result_ik = 0;
                for (std::size_t j = 0; j < input1.size(2); j++) {
                    result_ik += (input1[batch, i, j] * input2[j, k]);
                }

                REQUIRE_THAT((output[batch, i, k]), Catch::Matchers::WithinAbs(result_ik, 0.0001));
            }
        }
    }
}


TEST_CASE("Matmul large 2d", "[kernel::bmm]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::bmm<float> mm(gpu0);

    auto input1 = shared_tensor(full<float>({8, 2048}, 2.0));
    auto input2 = shared_tensor(full<float>({2048, 128256}, 1.0));

    BENCHMARK("multiply 128256 elements")
    {
        auto output = mm(input1, input2).get();

        REQUIRE(output.dim() == 2);
        REQUIRE(output.size(0) == 8);
        REQUIRE(output.size(1) == 128256);
    };

    // std::cout << output << std::endl;
    // for (auto it = output.begin(); it != output.end(); ++it) {
    //     REQUIRE_THAT(*it, Catch::Matchers::WithinAbs(4096.0, 1e-5));
    // }
}
