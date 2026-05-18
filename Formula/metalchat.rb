class Metalchat < Formula
  desc "Llama inference for Apple Silicon"
  homepage "https://metalchat.readthedocs.org"
  license "GPL-3.0-or-later"
  head "file://#{Pathname.new(__dir__).parent}"

  depends_on "conan@2" => :build
  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "openssl@3"
  depends_on "curl"

  def install
    build_path = Pathname.new(__dir__).parent

    build_args = %W[
      --build=missing
      --output-folder=build
      --conf tools.build:skip_test=True
      --settings build_type=Release
      --options use_system_libs=True
    ]

    system "conan", "profile", "detect"
    system "conan", "build", *build_args, build_path

    bin.install "#{build_path}/build/build/Release/metalchat"
    frameworks.install "#{build_path}/build/build/Release/MetalChat.framework"

    # Conan links the framework with an @rpath, here we override it with
    # an absolute path to the Homebrew frameworks path.
    MachO::Tools.add_rpath("#{bin}/metalchat", frameworks.to_s)

    system "codesign", "-s", "-", "-f", "#{bin}/metalchat"
  end
end
