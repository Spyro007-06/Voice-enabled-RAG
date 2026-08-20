$baseDir = $PSScriptRoot | Split-Path -Parent
$rawDir = Join-Path $baseDir "data\raw"
if (-not (Test-Path $rawDir)) { New-Item -ItemType Directory -Path $rawDir -Force }

$downloads = @(
    @{ name = "tamval.parquet"; url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/tamval.parquet" },
    @{ name = "telval.parquet"; url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/telval.parquet" },
    @{ name = "malval.parquet"; url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/malval.parquet" }
)

$jobs = @()
foreach ($d in $downloads) {
    $target = Join-Path $rawDir $d.name
    $url = $d.url
    Write-Host "Starting download: $($d.name) -> $target"
    $job = Start-Job -ScriptBlock {
        param($u, $t)
        curl.exe -L -C - -o $t $u --retry 5 --retry-delay 2 -s
    } -ArgumentList $url, $target
    $jobs += $job
}

Write-Host "Waiting for downloads to complete..."
$jobs | Wait-Job
Write-Host "All downloads finished!"
foreach ($d in $downloads) {
    $f = Get-Item (Join-Path $rawDir $d.name)
    Write-Host "$($d.name): $([math]::round($f.Length/1MB, 2)) MB"
}
