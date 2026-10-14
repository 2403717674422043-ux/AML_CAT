#!/bin/bash
set -e

# DDoSGuard Multi-service Entrypoint
MODE=${1:-"all"}

if [ "$MODE" = "api" ]; then
    echo "Starting FastAPI backend service on port 8000..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "dashboard" ]; then
    echo "Starting Streamlit dashboard on port 8501..."
    exec streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "Starting DDoSGuard dual service (API + Dashboard)..."
    # Start FastAPI in background
    uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API readiness
    echo "Waiting for API readiness..."
    sleep 3
    
    # Start Streamlit in foreground
    echo "Starting Streamlit Dashboard on port 8501..."
    export DDOSGUARD_API_URL="http://localhost:8000"
    streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
fi
