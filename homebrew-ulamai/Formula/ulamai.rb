class Ulamai < Formula
  desc "Ulam AI prover CLI for Lean 4"
  homepage "https://github.com/ulamai/ulamai"
  url "https://github.com/ulamai/ulamai/archive/refs/tags/v0.1.6.tar.gz"
  sha256 "e9520d843d4da4c44affc6b2fe4b08891283ae9cd0e668b687784b26493e6003"
  license "MIT"

  def install
    libexec.install Dir["*"]
    ENV["ULAM_VENV_DIR"] = (libexec/"venv").to_s
    system "bash", (libexec/"install.sh").to_s
    bin.install_symlink libexec/"venv/bin/ulam" => "ulam"
  end

  test do
    system "#{bin}/ulam", "--help"
  end
end
