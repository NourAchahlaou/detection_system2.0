# service_readiness_manager.py - Service Readiness Pattern with Circuit Breaker
import asyncio
import logging
import time
import aiohttp
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    UNKNOWN = "unknown"
    CHECKING = "checking"
    READY = "ready"
    DOWN = "down"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    max_failures: int = 3
    timeout_seconds: int = 60  # How long to keep circuit open
    retry_backoff_base: float = 1.0  # Base backoff time in seconds
    max_retry_attempts: int = 3

@dataclass
class ServiceHealthStatus:
    """Service health status tracking"""
    state: ServiceState = ServiceState.UNKNOWN
    last_check_time: float = 0
    consecutive_failures: int = 0
    last_success_time: float = 0
    circuit_open_until: float = 0
    total_checks: int = 0
    total_successes: int = 0
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

class ServiceReadinessManager:
    """
    Singleton service readiness manager with circuit breaker pattern.
    Ensures detection service is ready before establishing pubsub connections.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Configuration
        self.detection_service_url = "http://detection:8000"  # Adjust as needed
        self.circuit_config = CircuitBreakerConfig()
        
        # State tracking
        self.detection_service_status = ServiceHealthStatus()
        self._check_lock = asyncio.Lock()
        
        # HTTP session for health checks
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info("🔧 Service Readiness Manager initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def is_detection_service_ready(self, force_check: bool = False) -> bool:
        """
        Check if detection service is ready with circuit breaker pattern.
        
        Args:
            force_check: Force a new health check even if circuit is open
            
        Returns:
            bool: True if service is ready, False otherwise
        """
        async with self._check_lock:
            current_time = time.time()
            status = self.detection_service_status
            
            # Check if circuit breaker is open
            if not force_check and status.state == ServiceState.CIRCUIT_OPEN:
                if current_time < status.circuit_open_until:
                    logger.debug(f"🚫 Circuit breaker OPEN for detection service "
                               f"(opens in {status.circuit_open_until - current_time:.1f}s)")
                    return False
                else:
                    # Circuit breaker timeout expired, allow one retry
                    logger.info("🔄 Circuit breaker timeout expired, allowing retry")
                    status.state = ServiceState.UNKNOWN
            
            # If we have a recent successful check, return it
            if (not force_check and 
                status.state == ServiceState.READY and 
                current_time - status.last_success_time < 30):  # Cache for 30 seconds
                return True
            
            # Perform health check
            return await self._perform_health_check()
    
    async def _perform_health_check(self) -> bool:
        """Perform actual health check with exponential backoff"""
        status = self.detection_service_status
        status.state = ServiceState.CHECKING
        status.total_checks += 1
        
        for attempt in range(self.circuit_config.max_retry_attempts):
            try:
                start_time = time.time()
                
                session = await self._get_session()
                
                # Check the /redis/ready endpoint
                ready_url = f"{self.detection_service_url}/redis/ready"
                
                async with session.get(ready_url) as response:
                    response_time = (time.time() - start_time) * 1000
                    status.response_time_ms = response_time
                    status.last_check_time = time.time()
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Verify the service is truly ready
                        if self._validate_readiness_response(response_data):
                            return await self._handle_success()
                        else:
                            logger.warning(f"⚠️ Detection service responded 200 but not fully ready: {response_data}")
                            raise Exception(f"Service not fully ready: {response_data}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Health check failed with status {response.status}: {error_text}")
            
            except Exception as e:
                logger.warning(f"❌ Detection service health check attempt {attempt + 1} failed: {e}")
                
                if attempt < self.circuit_config.max_retry_attempts - 1:
                    # Exponential backoff
                    backoff_time = self.circuit_config.retry_backoff_base * (2 ** attempt)
                    logger.info(f"⏳ Retrying in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
                else:
                    # All attempts failed
                    return await self._handle_failure(str(e))
        
        return False
    
    def _validate_readiness_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate that the detection service is truly ready"""
        try:
            # Check required fields in response
            required_checks = [
                response_data.get('processor_initialized', False),
                response_data.get('redis_connected', False),
                response_data.get('pubsub_ready', False),
                response_data.get('status') == 'ready'
            ]
            
            all_ready = all(required_checks)
            
            if not all_ready:
                logger.debug(f"🔍 Readiness validation failed. Response: {response_data}")
            
            return all_ready
            
        except Exception as e:
            logger.error(f"❌ Error validating readiness response: {e}")
            return False
    
    async def _handle_success(self) -> bool:
        """Handle successful health check"""
        current_time = time.time()
        status = self.detection_service_status
        
        status.state = ServiceState.READY
        status.last_success_time = current_time
        status.consecutive_failures = 0
        status.total_successes += 1
        status.error_message = None
        
        logger.info(f"✅ Detection service is READY "
                   f"(response time: {status.response_time_ms:.1f}ms, "
                   f"success rate: {status.total_successes}/{status.total_checks})")
        
        return True
    
    async def _handle_failure(self, error_message: str) -> bool:
        """Handle failed health check with circuit breaker logic"""
        current_time = time.time()
        status = self.detection_service_status
        
        status.consecutive_failures += 1
        status.error_message = error_message
        status.last_check_time = current_time
        
        # Open circuit breaker if too many failures
        if status.consecutive_failures >= self.circuit_config.max_failures:
            status.state = ServiceState.CIRCUIT_OPEN
            status.circuit_open_until = current_time + self.circuit_config.timeout_seconds
            
            logger.error(f"💥 CIRCUIT BREAKER OPEN for detection service "
                        f"({status.consecutive_failures} consecutive failures). "
                        f"Will retry after {self.circuit_config.timeout_seconds}s")
        else:
            status.state = ServiceState.DOWN
            logger.warning(f"⚠️ Detection service DOWN "
                         f"({status.consecutive_failures}/{self.circuit_config.max_failures} failures)")
        
        return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status for monitoring"""
        status = self.detection_service_status
        current_time = time.time()
        
        return {
            'detection_service': {
                'state': status.state.value,
                'is_ready': status.state == ServiceState.READY,
                'consecutive_failures': status.consecutive_failures,
                'total_checks': status.total_checks,
                'total_successes': status.total_successes,
                'success_rate_percent': (status.total_successes / status.total_checks * 100) if status.total_checks > 0 else 0,
                'last_check_ago_seconds': current_time - status.last_check_time if status.last_check_time > 0 else None,
                'last_success_ago_seconds': current_time - status.last_success_time if status.last_success_time > 0 else None,
                'circuit_open_until': status.circuit_open_until if status.state == ServiceState.CIRCUIT_OPEN else None,
                'response_time_ms': status.response_time_ms,
                'error_message': status.error_message
            },
            'circuit_breaker': {
                'max_failures': self.circuit_config.max_failures,
                'timeout_seconds': self.circuit_config.timeout_seconds,
                'is_open': status.state == ServiceState.CIRCUIT_OPEN,
                'time_until_retry': max(0, status.circuit_open_until - current_time) if status.state == ServiceState.CIRCUIT_OPEN else 0
            }
        }
    
    async def wait_for_service_ready(self, timeout_seconds: int = 60) -> bool:
        """
        Wait for detection service to be ready with timeout.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            bool: True if service became ready, False if timeout
        """
        logger.info(f"⏳ Waiting for detection service to be ready (timeout: {timeout_seconds}s)...")
        
        start_time = time.time()
        check_interval = 2.0  # Check every 2 seconds
        
        while time.time() - start_time < timeout_seconds:
            if await self.is_detection_service_ready():
                elapsed = time.time() - start_time
                logger.info(f"✅ Detection service ready after {elapsed:.1f}s")
                return True
            
            await asyncio.sleep(check_interval)
        
        logger.error(f"❌ Detection service not ready after {timeout_seconds}s timeout")
        return False
    
    async def force_circuit_breaker_reset(self):
        """Force reset the circuit breaker (for admin/debug use)"""
        async with self._check_lock:
            status = self.detection_service_status
            status.state = ServiceState.UNKNOWN
            status.consecutive_failures = 0
            status.circuit_open_until = 0
            status.error_message = None
            
            logger.info("🔄 Circuit breaker forcefully reset")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()

# Global singleton instance
service_readiness_manager = ServiceReadinessManager()