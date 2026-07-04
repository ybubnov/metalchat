class Metalchat < Formula
  def self.source_path
    source_uri = URI.parse(url)
    Pathname.new(source_uri.path).parent
  end

  bottle do
    root_url "https://ghcr.io/v2/ybubnov/stable"
    sha256 cellar: :any, arm64_tahoe:   "ffc308f6fecdbdb5592b8ee4a8b9e904ff63c6b8c14a7d1ae7d0fb9c01b94bef"
    sha256 cellar: :any, arm64_sequoia: "c5eabb449c0c86c396ae36aa84d3f77179ddc0a48436d031a5e17cab949c1680"
  end

  def source_path
    Metalchat::source_path
  end

  def self.resolve_version
    File.read(source_path / "version.txt").strip
  end

  def self.resolve_url
    URI::File.build(path: __dir__).to_s
  end

  desc "Llama inference for Apple Silicon"
  homepage "https://metalchat.readthedocs.org"
  license "GPL-3.0-or-later"
  url resolve_url
  version resolve_version

  depends_on "conan@2" => :build
  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "openssl@3"
  depends_on "curl"

  def install
    build_path = buildpath/"build"

    build_args = %W[
      --build=missing
      --conf tools.build:skip_test=True
      --settings build_type=Release
      --options use_system_libs=True
    ]

    cp_r source_path, build_path
    system "conan", "profile", "detect"
    system "conan", "build", *build_args, build_path

    bin.install build_path/"build/Release/metalchat"
    frameworks.install build_path/"build/Release/MetalChat.framework"

    # Conan links the framework with an @rpath, here we override it with
    # an absolute path to the Homebrew frameworks path.
    MachO::Tools.add_rpath("#{bin}/metalchat", frameworks.to_s)

    system "codesign", "-s", "-", "-f", "#{bin}/metalchat"
  end
end
