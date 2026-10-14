class Metalchat < Formula
  def self.source_path
    source_uri = URI.parse(url)
    Pathname.new(source_uri.path).parent
  end

  bottle do
    root_url "https://ghcr.io/v2/ybubnov/stable"
    sha256 cellar: :any, arm64_golden_gate: "52fd8192fc2accaa308157b7dd0e24ba62e6227c1f63049672d2050d7e0089bb"
    sha256 cellar: :any, arm64_tahoe:       "6fb06e8218dfa5b5c0e96e5440c5fae6b47d1c74e252cc4295a867151ff40a96"
    sha256 cellar: :any, arm64_sequoia:     "801be42c881dad0a2253b7439c0610a1dbf3a5cbb676f408978e1ab7dcdced69"
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
