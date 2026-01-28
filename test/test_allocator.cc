// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/kernel.h>


using namespace metalchat;


TEST_CASE("Allocate hardware container", "[allocator]")
{
    hardware_accelerator gpu0;
    auto alloc = hardware_memory_allocator(gpu0.get_metal_device());

    auto b = alloc.allocate(10);
    REQUIRE(b != nullptr);
}


TEST_CASE("Resident allocator", "[allocator]")
{
    hardware_accelerator gpu0;
    auto alloc0 = hardware_memory_allocator(gpu0.get_metal_device());
    auto alloc1 = hardware_resident_allocator(alloc0, gpu0.get_metal_device());

    auto b = alloc1.allocate(10);
    REQUIRE(b != nullptr);
}


TEST_CASE("Allocator random memory container", "[allocator]")
{
    random_memory_allocator<void> alloc;
    auto container0 = alloc.allocate(3 * sizeof(std::size_t));

    REQUIRE(container0);
    REQUIRE(container0->size() == 24);
    REQUIRE(container0->data() != nullptr);

    auto container1 = container0;
    REQUIRE(container1->size() == 24);
    REQUIRE(container1->data() != nullptr);
}


TEST_CASE("Allocator offset", "[allocator]")
{
    random_memory_allocator<std::size_t> alloc;
    auto container0 = alloc.allocate(10);
    REQUIRE(container0->size() == 10 * sizeof(std::size_t));

    for (std::size_t i = 0; i < 10; i++) {
        container0->data()[i] = i;
    }

    for (std::size_t i = 0; i < 10; i++) {
        REQUIRE(container0->data()[i] == i);
    }
}
