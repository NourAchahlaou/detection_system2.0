import time
from typing import Dict, List, Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.service.circuitBreaker import CircuitBreaker

# Create router for monitoring dashboard
monitor_router = APIRouter(
    prefix="/monitor",
    tags=["Monitoring"],
    responses={404: {"description": "Not found"}},
)

# Create Jinja2 templates instance - adjust the directory as needed
templates = Jinja2Templates(directory="app/templates")

# Global registry of circuit breakers for monitoring
circuit_breaker_registry: Dict[str, CircuitBreaker] = {}

def register_circuit_breaker(circuit_breaker: CircuitBreaker) -> None:
    """Register a circuit breaker for monitoring"""
    circuit_breaker_registry[circuit_breaker.name] = circuit_breaker

@monitor_router.get("/circuit-breakers", response_class=HTMLResponse)
async def monitor_circuit_breakers(request: Request):
    """Display a monitoring dashboard for all circuit breakers"""
    breaker_statuses = []
    
    for name, cb in circuit_breaker_registry.items():
        last_failure_time_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", 
            time.localtime(cb.last_failure_time)
        ) if cb.last_failure_time > 0 else "Never"
        
        breaker_statuses.append({
            "name": name,
            "state": cb.current_state,
            "failure_count": cb.failure_count,
            "last_failure_time": last_failure_time_str,
            "recovery_timeout": cb.recovery_timeout,
            "failure_threshold": cb.failure_threshold
        })
    
    # Sort by state (OPEN first, then HALF_OPEN, then CLOSED)
    state_priority = {"OPEN": 0, "HALF_OPEN": 1, "CLOSED": 2}
    breaker_statuses.sort(key=lambda x: state_priority.get(x["state"], 3))
    
    return templates.TemplateResponse(
        "circuit_breakers.html",
        {"request": request, "breakers": breaker_statuses}
    )

@monitor_router.post("/reset-circuit-breaker/{name}")
async def reset_circuit_breaker(name: str):
    """Reset a specific circuit breaker"""
    if name not in circuit_breaker_registry:
        return {"status": "error", "message": f"Circuit breaker {name} not found"}
    
    circuit_breaker_registry[name].reset()
    return {"status": "success", "message": f"Circuit breaker {name} reset successfully"}

# Function to reset all circuit breakers
@monitor_router.post("/reset-all-circuit-breakers")
async def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    for name, cb in circuit_breaker_registry.items():
        cb.reset()
    
    return {"status": "success", "message": f"All {len(circuit_breaker_registry)} circuit breakers reset successfully"}