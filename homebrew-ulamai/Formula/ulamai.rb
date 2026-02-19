class Ulamai < Formula
  desc "Ulam AI prover CLI for Lean 4"
  homepage "https://github.com/ulamai/ulamai"
  url "https://github.com/ulamai/ulamai/archive/refs/tags/v0.1.8.tar.gz"
  sha256 "8b9fdce5bec0d824684258b22fae0cf4efbeb505a70db8f0d0e7f0d40fe01faf"
  license "MIT"

  depends_on "python@3.12"

  def install
    python = Formula["python@3.12"].opt_bin/"python3.12"
    ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    ENV["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    system python, "-m", "pip", "install", ".", "--prefix=#{prefix}"
  end

  test do
    system "#{bin}/ulam", "--help"
  end
end
