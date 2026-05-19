class Metalchat < Formula
  desc "Llama inference for Apple Silicon"
  homepage "https://metalchat.readthedocs.org"
  license "GPL-3.0-or-later"
  url "file://#{source_path}"

  depends_on "conan@2" => :build
  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "openssl@3"
  depends_on "curl"

  def version
    Version.new(File.read(source_path/"version.txt").strip)
  end

  def source_path
    Pathname.new(__dir__).parent
  end

  def install
    build_path = buildpath/"build"

    build_args = %W[
      --build=missing
      --conf tools.build:skip_test=True
      --settings build_type=Release
      --options use_system_libs=True
    ]

    cp_r sourcepath, build_path
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
