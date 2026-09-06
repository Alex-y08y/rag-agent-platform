# RAG Agent Platform - Startup Script (PowerShell)
# Right-click -> Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File start.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Show-Menu {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "       Enterprise RAG + Agent Knowledge Platform" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Docker Mode Start (Recommended - Full Stack)"
    Write-Host "  [2] Docker Mode Stop"
    Write-Host "  [3] Docker Mode - View Backend Logs"
    Write-Host "  [4] Local Dev Mode Start (Backend + Frontend)"
    Write-Host "  [5] Init Database + Import Demo Data"
    Write-Host "  [6] Run Tests"
    Write-Host "  [7] Exit"
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Start-DockerMode {
    Write-Host ""
    Write-Host "[1/4] Checking .env config..." -ForegroundColor Yellow

    if (-not (Test-Path ".env")) {
        Write-Host ".env not found, copying from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host ".env created. Please fill in LLM API Key in notepad." -ForegroundColor Yellow
        Read-Host "Press Enter to open .env for editing"
        notepad ".env"
    }
    Write-Host ".env ready" -ForegroundColor Green

    Write-Host ""
    Write-Host "[2/4] Building and starting Docker services..." -ForegroundColor Yellow
    Write-Host "First run may take 5-15 minutes to download images." -ForegroundColor Gray
    Write-Host ""

    docker-compose up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] Docker start failed. Make sure Docker Desktop is running." -ForegroundColor Red
        Read-Host "Press Enter to return"
        return
    }

    Write-Host ""
    Write-Host "[3/4] Waiting for backend to be ready (30s)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30

    Write-Host ""
    Write-Host "[4/4] Initializing database and importing demo data..." -ForegroundColor Yellow
    docker-compose exec -T backend python -c "from app.core.database import init_db; init_db(); print('DB initialized')"
    docker-compose exec -T backend python ../scripts/seed_demo_data.py

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Startup Complete!" -ForegroundColor Green
    Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor White
    Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Green
    Read-Host "Press Enter to return"
}

function Stop-DockerMode {
    Write-Host ""
    Write-Host "Stopping all Docker services..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "All services stopped" -ForegroundColor Green
    Read-Host "Press Enter to return"
}

function Show-DockerLogs {
    Write-Host ""
    Write-Host "Backend logs (Ctrl+C to exit)..." -ForegroundColor Yellow
    Write-Host ""
    docker-compose logs -f backend
}

function Start-LocalDev {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Local Development Mode" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "[1/4] Checking Python virtual environment..." -ForegroundColor Yellow
    $venvPython = Join-Path $ProjectRoot "backend\venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating venv with Python 3.12..." -ForegroundColor Yellow
        py -3.12 -m venv backend\venv
        Write-Host "venv created" -ForegroundColor Green
    } else {
        Write-Host "venv exists" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "[2/4] Installing Python dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install -r backend\requirements.txt -q
    Write-Host "Python deps installed" -ForegroundColor Green

    Write-Host ""
    Write-Host "[3/4] Installing frontend dependencies..." -ForegroundColor Yellow
    $nodeModules = Join-Path $ProjectRoot "frontend\node_modules"
    if (-not (Test-Path $nodeModules)) {
        Push-Location frontend
        npm install
        Pop-Location
    } else {
        Write-Host "node_modules exists, skipping" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "[4/4] Initializing database..." -ForegroundColor Yellow
    & $venvPython -c "from app.core.database import init_db; init_db(); print('Database initialized')"

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Starting services..." -ForegroundColor Green
    Write-Host "  Backend:  http://localhost:8000  (API docs: /docs)" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "  Each runs in a new window. Close window to stop." -ForegroundColor Gray
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""

    $backendCmd = 'cd /d "' + $ProjectRoot + '\backend" && venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000'
    Start-Process cmd -ArgumentList "/k", $backendCmd
    Start-Sleep -Seconds 3

    $frontendCmd = 'cd /d "' + $ProjectRoot + '\frontend" && npm run dev'
    Start-Process cmd -ArgumentList "/k", $frontendCmd

    Write-Host "Backend and frontend started in new windows" -ForegroundColor Green
    Write-Host "To import demo data, select [5] from menu." -ForegroundColor Yellow
    Read-Host "Press Enter to return"
}

function Invoke-SeedData {
    Write-Host ""
    Write-Host "Initializing database and importing demo data..." -ForegroundColor Yellow
    Write-Host ""

    $venvPython = Join-Path $ProjectRoot "backend\venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        Push-Location backend
        & $venvPython -c "from app.core.database import init_db; init_db()"
        & $venvPython ..\scripts\seed_demo_data.py
        Pop-Location
    } else {
        Write-Host "Local venv not found, trying Docker..." -ForegroundColor Yellow
        docker-compose exec -T backend python -c "from app.core.database import init_db; init_db()"
        docker-compose exec -T backend python ../scripts/seed_demo_data.py
    }

    Write-Host ""
    Write-Host "Demo data imported" -ForegroundColor Green
    Read-Host "Press Enter to return"
}

function Invoke-RunTests {
    Write-Host ""
    Write-Host "Running tests..." -ForegroundColor Yellow
    Write-Host ""

    $venvPython = Join-Path $ProjectRoot "backend\venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        Push-Location backend
        & $venvPython -m pytest tests/ -v --tb=short
        Pop-Location
    } else {
        Write-Host "Local venv not found. Use [4] Local Dev Mode first." -ForegroundColor Red
    }

    Write-Host ""
    Read-Host "Press Enter to return"
}

# Main loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Select (1-7)"

    switch ($choice) {
        "1" { Start-DockerMode }
        "2" { Stop-DockerMode }
        "3" { Show-DockerLogs }
        "4" { Start-LocalDev }
        "5" { Invoke-SeedData }
        "6" { Invoke-RunTests }
        "7" {
            Write-Host ""
            Write-Host "Goodbye!" -ForegroundColor Cyan
            exit
        }
        default {
            Write-Host "Invalid selection" -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
}
