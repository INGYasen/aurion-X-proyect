# Compila docs/AURION_X_APEX_ARCHITECTURE.tex -> PDF via Docker TeX Live
$ErrorActionPreference = "Stop"
$docs = Join-Path $PSScriptRoot "..\docs" | Resolve-Path
$tex = "AURION_X_APEX_ARCHITECTURE.tex"
$pdf = "AURION_X_APEX_ARCHITECTURE.pdf"

Write-Host "Compilando $tex en $docs ..."

docker run --rm `
  -v "${docs}:/work" `
  -w /work `
  texlive/texlive:latest `
  sh -c "pdflatex -interaction=nonstopmode $tex && pdflatex -interaction=nonstopmode $tex"

$out = Join-Path $docs $pdf
if (Test-Path $out) {
    $size = (Get-Item $out).Length
    Write-Host "OK: $out ($([math]::Round($size/1KB)) KB)"
} else {
    Write-Error "No se genero el PDF. Revisa ${docs}\AURION_X_APEX_ARCHITECTURE.log"
    exit 1
}
