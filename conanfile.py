from pathlib import Path
from shutil import copytree

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools import apple
from conan.tools import files
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class MetalChat(ConanFile):
    name = "metalchat"
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
        "build_executable": [True, False],
    }

    default_options = {
        "shared": True,
        "fPIC": True,
        "build_executable": True,

        # Default options of the dependent packages.
        "catch2/*:shared": True,
        "metal-cpp/*:shared": True,
        "jsoncons/*:shared": False,
        "pcre2/*:shared": False,
        "rapidhash/*:shared": False,
        "mbits-mstch/*:shared": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "LICENSE",
        "include/*",
        "kernel/*",
        "src/*",
        "test/*",
        "module.modulemap",
    )

    def set_version(self):
        self.version = files.load(self, "version.txt").strip()

    def validate(self):
        if not apple.is_apple_os(self):
            raise ConanInvalidConfiguration(
                f"{self.name} supports only Apple operating systems"
            )

    def requirements(self):
        self.requires("cppcodec/0.2")
        self.requires("mbits-mstch/1.0.4")
        self.requires("metal-cpp/15.2")
        self.requires("rapidhash/3.0")
        self.requires("jsoncons/1.3.0")
        self.requires("pcre2/[>=10.30 <11.0]")

        if self.options.build_executable:
            self.requires("argparse/[>=3.2 <4.0.0]")
            self.requires("libcurl/[>=8.17.0 <9.0.0]")
            self.requires("keychain/[>=1.3.0 <2.0.0]")
            self.requires("replxx/[>=0.0.4 <1.0.0]")
            self.requires("toml11/[>=4.0.0 <5.0.0]")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.31.0]")
        self.tool_requires("ninja/[~1.13.2]")
        self.test_requires("catch2/[>=3.7.0 <4.0.0]")

    def generate(self):
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

        cmake_toolchain = CMakeToolchain(self, generator="Ninja")
        cmake_toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def layout(self):
        cmake_layout(self)

    def package(self):
        licenses_dst = str(Path(self.package_folder) / "licenses")
        files.copy(self, pattern="LICENSE", src=self.source_folder, dst=licenses_dst)

        framework_src = str(Path(self.build_folder) / "MetalChat.framework")
        frameworks_dst = str(Path(self.package_folder) / "Frameworks/MetalChat.framework")
        copytree(framework_src, frameworks_dst, symlinks=True, dirs_exist_ok=True)

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "metalchat")
        self.cpp_info.set_property("cmake_target_name", "metalchat::metalchat")
        self.cpp_info.set_property("pkg_config_name", "metalchat")

        self.cpp_info.includedirs = ["Frameworks/MetalChat.framework/Headers"]
        self.cpp_info.frameworkdirs = ["Frameworks"]
        self.cpp_info.frameworks = ["MetalChat"]
