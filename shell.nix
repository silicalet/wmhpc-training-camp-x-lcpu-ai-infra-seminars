{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  inherit (pkgs) lib;
  cuda = pkgs.cudaPackages.cudatoolkit;
in
pkgs.mkShell {
  packages = with pkgs; [
    cuda
    cmake
    git
    gnumake
    ninja
    pkg-config
    python312
    uv
  ];

  CUDA_HOME = "${cuda}";
  CUDA_PATH = "${cuda}";
  UV_PYTHON = "${pkgs.python312}/bin/python";
  UV_PYTHON_DOWNLOADS = "never";

  shellHook = ''
    export LD_LIBRARY_PATH="${
      lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ]
    }:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo "CUDA: $(nvcc --version | tail -n1)"
    echo "Python packages: cd assignment01 && uv sync --extra tilelang"
  '';
}
