# nix build --impure --expr 'let pkgs = import (builtins.getFlake "nixpkgs").outPath {}; in import ./assignment02/handout/texlive.nix { inherit pkgs; }' --out-link /tmp/assignment02-texlive
{ pkgs }:
pkgs.texlive.combine {
  inherit (pkgs.texlive)
    scheme-small ctex fandol xecjk fontspec tcolorbox environ trimspaces
    enumitem listings booktabs multirow etoolbox geometry xcolor hyperref
    amsmath amsfonts fancyvrb framed upquote pgf fvextra tikzfill pdfcol
    listingsutf8;
}
