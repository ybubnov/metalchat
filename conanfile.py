from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class MetalChat(ConanFile):
    name = "metalchat"
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
        "testing": [True, False],
    }

    default_options = {
        "shared": True,
        "fPIC": True,
        "testing": True,

        "catch2/*:shared": True,
        "metal-cpp/*:shared": True,
        "jsoncons/*:shared": False,
        "pcre2/*:shared": False,
        "rapidhash/*:shared": False,
        "mbits-mstch/*:shared": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "include/*",
        "kernel/*",
        "src/*",
        "test/*",
    )

    def requirements(self):
        self.requires("cppcodec/0.2")
        self.requires("mbits-mstch/1.0.4")
        self.requires("metal-cpp/15.2")
        self.requires("rapidhash/3.0")
        self.requires("jsoncons/1.3.0")
        self.requires("pcre2/10.44")

    def build_requirements(self):
        self.tool_requires("cmake/4.1.0")
        self.tool_requires("ninja/1.13.2")
        self.test_requires("catch2/3.7.1")

    def generate(self):
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

        cmake_toolchain = CMakeToolchain(self, generator="Ninja")
        cmake_toolchain.variables["BUILD_TESTING"] = "ON" if self.options.testing else "OFF"
        cmake_toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

        if self.options.testing and can_run(self):
            cmake.test()

    def layout(self):
        cmake_layout(self)
