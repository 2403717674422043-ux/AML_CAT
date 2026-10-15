#!/bin/bash
set -e

# DDoSGuard Multi-service Entrypoint
MODE=${1:-"all"}

if [ "$MODE" = "api" ]; then
    echo "Starting FastAPI backend service on port 8000..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "dashboard" ]; then
    DASHBOARD_PORT=${PORT:-8501}
    echo "Starting Streamlit dashboard on port ${DASHBOARD_PORT}..."
    exec streamlit run dashboard/app.py --server.port "${DASHBOARD_PORT}" --server.address 0.0.0.0 --server.headless true
else
    echo "Starting DDoSGuard dual service (API + Dashboard)..."
    # Start FastAPI in background
    uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API readiness
    echo "Waiting for API readiness..."
    sleep 3
    
    # Start Streamlit in foreground
    DASHBOARD_PORT=${PORT:-8501}
    echo "Starting Streamlit Dashboard on port ${DASHBOARD_PORT}..."
    export DDOSGUARD_API_URL="http://localhost:8000"
    exec streamlit run dashboard/app.py --server.port "${DASHBOARD_PORT}" --server.address 0.0.0.0 --server.headless true
fi
