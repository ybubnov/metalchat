class Metalchat < Formula
  def self.source_path
    source_uri = URI.parse(url)
    Pathname.new(source_uri.path).parent
  end

  bottle do
    root_url "https://ghcr.io/v2/ybubnov/stable"
    sha256 cellar: :any, arm64_golden_gate: "178a2a8177d6eb45245f3ac96b2c7d6f60a12aa7bec290d5cdcff0998c5eec2d"
    sha256 cellar: :any, arm64_tahoe:       "2319cb17407418949d01e02c16a27b641b2a773885827539bf89fce96abc540c"
    sha256 cellar: :any, arm64_sequoia:     "0a26951134439bc17244c64c5194c015f9281b83213b965911d4bfad84e1c7c2"
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
