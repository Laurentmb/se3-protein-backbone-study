param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [int]$Length = 100,
    [int]$Seed = 123,
    [string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $WorkspaceRoot 'work\runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The prepared runtime was not found. See README.md for a fresh setup.' }
$experimentPath = Join-Path $WorkspaceRoot ('outputs\se3-rerun-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
foreach ($modelName in @('framediff','frameflow')) {
    & $pythonPath -u (Join-Path $PSScriptRoot 'generate_backbone.py') --model $modelName --repo (Join-Path $WorkspaceRoot ('work\' + $modelName)) --output (Join-Path $experimentPath $modelName) --length $Length --seed $Seed --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Generation failed for $modelName" }
}
& $pythonPath (Join-Path $PSScriptRoot 'validate_backbones.py') --root $experimentPath --length $Length
if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }
Write-Output "Results saved to $experimentPath"

