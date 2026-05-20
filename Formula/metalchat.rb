class Metalchat < Formula
  desc "Llama inference for Apple Silicon"
  homepage "https://metalchat.readthedocs.org"
  license "GPL-3.0-or-later"
  head "https://github.com/ybubnov/metalchat.git", branch: "main"

  bottle do
    root_url "https://ghcr.io/v2/ybubnov/stable"
    sha256 cellar: :any, arm64_tahoe:   "c12730ec5cf70e0fed92562300f1cda6437235e65db40bc87e78ef11f44dd474"
    sha256 cellar: :any, arm64_sequoia: "f5898f9167651c9de637140a666b853168807693a9332a38fead71563783e3bd"
  end

  depends_on "conan@2" => :build
  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "openssl@3"
  depends_on "curl"

  def install
    build_args = %W[
      --build=missing
      --output-folder=build
      --conf tools.build:skip_test=True
      --settings build_type=Release
      --options use_system_libs=True
    ]

    system "conan", "profile", "detect"
    system "conan", "build", *build_args

    bin.install "build/build/Release/metalchat"
    frameworks.install "build/build/Release/MetalChat.framework"

    # Conan links the framework with an @rpath, here we override it with
    # an absolute path to the Homebrew frameworks path.
    MachO::Tools.add_rpath("#{bin}/metalchat", frameworks.to_s)

    system "codesign", "-s", "-", "-f", "#{bin}/metalchat"
  end
end
