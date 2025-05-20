from enum import Enum
import time
from typing import Callable, TypeVar
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("camera_circuit_breaker")

# Define return type for the circuit breaker
T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = 'CLOSED' 
    OPEN = 'OPEN'      
    HALF_OPEN = 'HALF_OPEN'  
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for hardware resilience.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        name: str = "default"
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before trying to recover (half-open state)
            name: Name of this circuit breaker instance for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    def execute(self, func: Callable[..., T], fallback: Callable[..., T] = None, *args, **kwargs) -> T:
        """
        Execute the function with circuit breaker protection.
        
        Args:
            func: The function to execute
            fallback: Function to call when circuit is open
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the function or fallback
            
        Raises:
            Exception: If circuit is open and no fallback is provided
        """
        if self._is_open():
            if fallback:
                logger.info(f"Circuit '{self.name}' is OPEN, using fallback")
                return fallback(*args, **kwargs)
            else:
                logger.warning(f"Circuit '{self.name}' is OPEN, no fallback provided")
                raise Exception(f"Circuit '{self.name}' is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            if fallback:
                logger.info(f"Function failed, using fallback for circuit '{self.name}'")
                return fallback(*args, **kwargs)
            raise
    
    def _is_open(self) -> bool:
        """Check if the circuit is open and handle state transitions."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Recovery timeout elapsed for circuit '{self.name}', transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                return False
            return True
        return False
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Successful execution in HALF_OPEN state for circuit '{self.name}', resetting to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle execution failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Execution failed for circuit '{self.name}': {str(exception)}, failure count: {self.failure_count}")
        
        if self.state == CircuitState.HALF_OPEN or (
            self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold
        ):
            logger.warning(f"Opening circuit '{self.name}' due to failures")
            self.state = CircuitState.OPEN
    
    @property
    def current_state(self) -> str:
        """Get the current state of the circuit breaker."""
        return self.state.value

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")