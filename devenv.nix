{ pkgs, lib, ... }:
let
  cuda = pkgs.cudaPackages.cudatoolkit;
in
{
  name = "wmhpc-ai-infra";

  packages = [
    cuda
    pkgs.cmake
    pkgs.git
    pkgs.gnumake
    pkgs.ninja
    pkgs.pkg-config
    pkgs.pandoc
    pkgs.zlib
  ];

  # Interpreter + uv only. Each assignment owns its pyproject.toml / uv sync.
  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv.enable = true;
  };

  env.CUDA_HOME = "${cuda}";
  env.CUDA_PATH = "${cuda}";
  env.UV_PYTHON_DOWNLOADS = "never";

  enterShell = ''
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
