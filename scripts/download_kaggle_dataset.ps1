$ErrorActionPreference = "Stop"

$dataset = "rabieelkharoua/predicting-manufacturing-defects-dataset"
$destination = "data/raw"

New-Item -ItemType Directory -Force -Path $destination | Out-Null
kaggle datasets download -d $dataset -p $destination --unzip

Write-Host "Dataset downloaded to $destination"
