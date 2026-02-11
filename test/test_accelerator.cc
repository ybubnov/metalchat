// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>


using namespace metalchat;


TEST_CASE("Create accelerator with wrong shader library", "[accelerator]")
{
    REQUIRE_THROWS_MATCHES(
        hardware_accelerator("some/nonexisting/file", 1), std::runtime_error,
        Catch::Matchers::Message("metal: library not found")
    );
}
