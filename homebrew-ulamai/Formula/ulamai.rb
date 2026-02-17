class Ulamai < Formula
  desc "Ulam AI prover CLI for Lean 4"
  homepage "https://github.com/ulamai/ulamai"
  url "https://raw.githubusercontent.com/ulamai/ulamai/v0.1.6/install.sh"
  sha256 "7eb552a8692b15c3f2810a2dba262a4e2e59ec11fe6398b9549c21e8a66efdbe"
  license "MIT"

  def install
    libexec.install "install.sh"
    ENV["ULAM_VENV_DIR"] = (libexec/"venv").to_s
    system "bash", "#{libexec}/install.sh"
    bin.install_symlink libexec/"venv/bin/ulam" => "ulam"
  end

  test do
    system "#{bin}/ulam", "--help"
  end
end
