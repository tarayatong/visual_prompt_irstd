$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repo 'outputs/core_ir_followup_20261117'
$python = Join-Path $env:USERPROFILE 'miniconda3/envs/sam/python.exe'
Set-Location -LiteralPath $repo
New-Item -ItemType Directory -Force -Path $output | Out-Null
$log = Join-Path $output 'dispatcher.log'
"Start $(Get-Date -Format o)" | Add-Content -LiteralPath $log
& $python -B -u -m scripts.run_core_ir_followup_queue *>> $log
$code = $LASTEXITCODE
"Exit $code at $(Get-Date -Format o)" | Add-Content -LiteralPath $log
exit $code
