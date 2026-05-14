<#
.SYNOPSIS
    TransitMind-Sogamoso - Script unificado de ejecucion.

.DESCRIPTION
    Ejecuta todo el pipeline (generacion, entrenamiento, evaluacion, pipeline completo)
    y levanta los servicios (MLflow UI + API) con un solo comando.

.PARAMETER Setup
    Instala dependencias (requirements.txt). Solo necesario la primera vez
    o cuando cambien las dependencias.

.PARAMETER PipelineOnly
    Ejecuta solo los pipelines sin levantar los servicios.

.PARAMETER ServicesOnly
    Levanta solo los servicios (MLflow UI + API) sin ejecutar pipelines.

.PARAMETER SkipPipeline
    Omite la ejecucion de pipelines y solo levanta servicios.

.PARAMETER Clean
    Limpia artefactos generados (datos, modelos, experimentos).

.EXAMPLE
    .\run.ps1                    # Ejecuta pipelines + levanta servicios
    .\run.ps1 -Setup             # Instala dependencias primero, luego ejecuta todo
    .\run.ps1 -PipelineOnly      # Solo ejecuta los pipelines
    .\run.ps1 -ServicesOnly      # Solo levanta MLflow UI + API
    .\run.ps1 -Clean             # Limpia artefactos generados
#>

param(
    [switch]$Setup,
    [switch]$PipelineOnly,
    [switch]$ServicesOnly,
    [switch]$SkipPipeline,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# -- Colores para la consola ------------------------------------------
function Write-Step  { param($msg) Write-Host "`n> $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red }

# -- Verificar que estamos en el directorio correcto ------------------
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path "requirements.txt")) {
    Write-Err "No se encontro requirements.txt. Ejecuta este script desde la raiz del proyecto."
    exit 1
}

# -- Clean ------------------------------------------------------------
if ($Clean) {
    Write-Step "Limpiando artefactos generados..."
    $paths = @(
        "data/processed/*.csv",
        "data/synthetic/generated_flows/*.csv",
        "models/timegan/checkpoints/*.pt",
        "experiments/mlruns",
        "experiments/pipeline_state.json"
    )
    foreach ($p in $paths) {
        $fullPath = Join-Path $projectRoot $p
        if (Test-Path $fullPath) {
            Remove-Item $fullPath -Recurse -Force
            Write-Ok "Eliminado: $p"
        }
    }
    # Limpiar __pycache__
    Get-ChildItem -Path $projectRoot -Directory -Recurse -Filter "__pycache__" | 
        ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
    if (Test-Path (Join-Path $projectRoot ".pytest_cache")) {
        Remove-Item (Join-Path $projectRoot ".pytest_cache") -Recurse -Force
    }
    Write-Ok "Limpieza completada."
    exit 0
}

# -- Setup (solo cuando se pide) --------------------------------------
if ($Setup) {
    Write-Step "Instalando dependencias..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Fallo la instalacion de dependencias."
        exit 1
    }
    Write-Ok "Dependencias instaladas correctamente."
}

# -- Pipelines --------------------------------------------------------
if (-not $ServicesOnly -and -not $SkipPipeline) {
    $pipelines = @(
        @{ Name = "Generacion de datos";   Script = "pipelines/pipeline_generate_data.py" },
        @{ Name = "Entrenamiento TimeGAN"; Script = "pipelines/pipeline_train_timegan.py" },
        @{ Name = "Evaluacion TSTR";       Script = "pipelines/pipeline_evaluate_tstr.py" },
        @{ Name = "Pipeline completo";     Script = "pipelines/pipeline_full.py" }
    )

    $totalSteps = $pipelines.Count
    $currentStep = 0

    foreach ($p in $pipelines) {
        $currentStep++
        Write-Step "[$currentStep/$totalSteps] $($p.Name)..."
        
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        python $p.Script
        $sw.Stop()

        if ($LASTEXITCODE -ne 0) {
            Write-Err "Fallo: $($p.Name)"
            Write-Err "Script: $($p.Script)"
            exit 1
        }
        Write-Ok "$($p.Name) completado en $([math]::Round($sw.Elapsed.TotalSeconds, 1))s"
    }

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  Todos los pipelines ejecutados con exito" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}

if ($PipelineOnly) {
    exit 0
}

# -- Servicios (MLflow UI + API) --------------------------------------
Write-Step "Levantando servicios..."

# Verificar si los puertos ya estan en uso
$port5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue

if ($port5000) {
    Write-Warn "Puerto 5000 ya en uso. MLflow UI podria estar ejecutandose."
}
if ($port8000) {
    Write-Warn "Puerto 8000 ya en uso. La API podria estar ejecutandose."
}

# Lanzar MLflow UI en background
Write-Step "Iniciando MLflow UI en puerto 5000..."
$mlflowJob = Start-Job -ScriptBlock {
    Set-Location $using:projectRoot
    mlflow ui --backend-store-uri file:./experiments/mlruns --port 5000
}
Write-Ok "MLflow UI iniciado (Job ID: $($mlflowJob.Id))"

# Lanzar API con uvicorn en foreground (para ver logs en tiempo real)
Write-Step "Iniciando API en puerto 8000..."
Write-Host ""
Write-Host "  +---------------------------------------------+" -ForegroundColor Magenta
Write-Host "  |  MLflow UI:  http://localhost:5000          |" -ForegroundColor Magenta
Write-Host "  |  API:        http://localhost:8000          |" -ForegroundColor Magenta
Write-Host "  |  API Docs:   http://localhost:8000/docs     |" -ForegroundColor Magenta
Write-Host "  |                                             |" -ForegroundColor Magenta
Write-Host "  |  Presiona Ctrl+C para detener todo          |" -ForegroundColor Magenta
Write-Host "  +---------------------------------------------+" -ForegroundColor Magenta
Write-Host ""

try {
    uvicorn src.layer1_timegan.api:app --host 0.0.0.0 --port 8000 --reload
}
finally {
    # Cuando se detiene la API, tambien detener MLflow
    Write-Step "Deteniendo servicios..."
    Stop-Job -Job $mlflowJob -ErrorAction SilentlyContinue
    Remove-Job -Job $mlflowJob -Force -ErrorAction SilentlyContinue
    Write-Ok "Todos los servicios detenidos."
}
