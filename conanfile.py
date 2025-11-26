from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.apple import XcodeToolchain, XcodeDeps


class MetalChat(ConanFile):
    name = "MetalChat"
    version = "1.0.0"
    package_type = "library"

    license = "GPL-3.0-or-later"
    author = "Yakau Bubnou (girokompass@gmail.com)"
    description = "Llama inference for Apple Devices"
    url = "https://github.com/ybubnov/metalchat"
    topics = ("deep-learning", "machine-learning", "neural-networks", "llama")

    settings = "os", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }

    default_options = {
        "shared": True,
        "fPIC": True,
        "catch2/*:shared": True,
        "metal-cpp/*:shared": True,
        "jsoncons/*:shared": False,
        "pcre2/*:shared": False,
        "rapidhash/*:shared": False,
        "mbits-mstch/*:shared": False,
    }

    def requirements(self):
        self.requires("catch2/3.7.1")
        self.requires("cppcodec/0.2")
        self.requires("mbits-mstch/1.0.4")
        self.requires("metal-cpp/15.2")
        self.requires("rapidhash/3.0")
        self.requires("jsoncons/1.3.0")
        self.requires("pcre2/10.44")

    def generate(self):
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

        cmake_toolchain = CMakeToolchain(self)
        cmake_toolchain.generate()

    def layout(self):
        cmake_layout(self, src_folder="src")
