#include <catch2/catch_test_macros.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/kernel.h>


using namespace metalchat;


TEST_CASE("Allocate hardware container", "[allocator]")
{
    hardware_accelerator gpu0;
    hardware_memory_allocator<float> alloc(gpu0.get_metal_device());

    auto b = alloc.allocate(10);
    REQUIRE(b != nullptr);
}


TEST_CASE("Resident allocator", "[allocator]")
{
    hardware_accelerator gpu0;
    auto alloc0 = hardware_memory_allocator<float>(gpu0.get_metal_device());
    auto alloc1 = hardware_resident_allocator(alloc0, gpu0.get_metal_device());

    auto b = alloc1.allocate(10);
    REQUIRE(b != nullptr);
}
