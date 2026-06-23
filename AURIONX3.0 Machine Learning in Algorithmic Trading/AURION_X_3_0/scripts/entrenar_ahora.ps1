# Entrena el modelo ML con todos los trades cerrados de PostgreSQL
Set-Location $PSScriptRoot\..
python -m core.train_model --config config/parameters.yaml --batch --limit 5000
Write-Host "Modelo guardado segun ml.model_path en parameters.yaml"
