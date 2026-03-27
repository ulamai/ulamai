class Ulamai < Formula
  include Language::Python::Virtualenv

  desc "Ulam AI prover CLI for Lean 4"
  homepage "https://github.com/ulamai/ulamai"
  url "https://github.com/ulamai/ulamai/archive/refs/tags/v0.2.11.tar.gz"
  sha256 "efaaf5e23b6fb6ca46515a180714dcb11bab28d48565531791df3aaa7460a38e"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/ulam", "--help"
  end
end
