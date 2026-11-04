$ErrorActionPreference = 'Stop'
$toolsDir = Join-Path $PSScriptRoot '.tools'
New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
$javaDir = Join-Path $toolsDir 'java'
if (-not (Test-Path (Join-Path $javaDir 'bin/java.exe'))) {
    $assets = Invoke-RestMethod 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jre&os=windows&vendor=eclipse'
    $package = $assets[0].binary.package
    $archive = Join-Path $toolsDir 'java.zip'
    Write-Host "Downloading Java: $($package.name)"
    Invoke-WebRequest -Uri $package.link -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.checksum.ToLowerInvariant()) {
        throw 'Java archive checksum mismatch'
    }
    $unpackDir = Join-Path $toolsDir 'java-unpack'
    Expand-Archive -LiteralPath $archive -DestinationPath $unpackDir -Force
    $javaRoot = Get-ChildItem -LiteralPath $unpackDir -Directory | Select-Object -First 1
    Move-Item -LiteralPath $javaRoot.FullName -Destination $javaDir
    $package | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $toolsDir 'java-download.json') -Encoding utf8
}

$modelDir = Join-Path $toolsDir 'vncorenlp'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
# Pin all files to the same upstream commit.
$commit = (Invoke-RestMethod 'https://api.github.com/repos/vncorenlp/VnCoreNLP/commits/master').sha
$files = @(
    'VnCoreNLP-1.2.jar', 'LICENSE.md',
    'models/wordsegmenter/vi-vocab', 'models/wordsegmenter/wordsegmenter.rdr',
    'models/postagger/vi-tagger', 'models/ner/vi-500brownclusters.xz',
    'models/ner/vi-ner.xz', 'models/ner/vi-pretrainedembeddings.xz',
    'models/dep/vi-dep.xz'
)
$manifest = foreach ($relative in $files) {
    $destination = Join-Path $modelDir $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Write-Host "Downloading $relative"
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/$commit/$relative" -OutFile $destination
    [PSCustomObject]@{
        file = $relative
        sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes = (Get-Item -LiteralPath $destination).Length
    }
}
@{ repository = 'https://github.com/vncorenlp/VnCoreNLP'; commit = $commit; files = @($manifest) } |
    ConvertTo-Json -Depth 5 | Set-Content (Join-Path $modelDir 'download-manifest.json') -Encoding utf8
& (Join-Path $javaDir 'bin/java.exe') -version
if ($LASTEXITCODE -ne 0) { throw 'Java verification failed' }
Write-Host 'Java and VnCoreNLP models installed in .tools.'
