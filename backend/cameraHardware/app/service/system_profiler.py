# system_profiler.py - Fixed version for Python 3.12+
import psutil
import subprocess
import time
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import platform
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SystemSpecs:
    """System specifications and performance metrics"""
    cpu_cores: int
    cpu_frequency_mhz: float
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_available: bool
    average_cpu_load: float
    system_platform: str
    recommended_mode: str
    performance_score: float
    timestamp: datetime

class SystemProfiler:
    """
    System profiler that determines optimal performance mode
    Fixed for Python 3.12+ compatibility - no longer uses GPUtil
    """
    
    def __init__(self):
        self.performance_history = []
        self.monitoring_duration = 10  # seconds to monitor for baseline
        self.high_performance_threshold = 75  # performance score threshold
        
    def get_system_specs(self) -> SystemSpecs:
        """Get comprehensive system specifications"""
        try:
            # CPU Information
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current if cpu_freq else 0
            
            # Memory Information
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            # CPU Load (sample over short period)
            cpu_loads = []
            for _ in range(5):
                cpu_loads.append(psutil.cpu_percent(interval=0.5))
            average_cpu_load = sum(cpu_loads) / len(cpu_loads)
            
            # GPU Information using nvidia-ml-py or nvidia-smi fallback
            gpu_info = self._get_gpu_info()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                cpu_cores, cpu_frequency, total_ram_gb, available_ram_gb,
                gpu_info['available'], gpu_info['memory_gb'], average_cpu_load, 
                gpu_info['cuda_available']
            )
            
            # Determine recommended mode
            recommended_mode = "high" if performance_score >= self.high_performance_threshold else "low"
            
            specs = SystemSpecs(
                cpu_cores=cpu_cores,
                cpu_frequency_mhz=cpu_frequency,
                total_ram_gb=round(total_ram_gb, 2),
                available_ram_gb=round(available_ram_gb, 2),
                gpu_available=gpu_info['available'],
                gpu_name=gpu_info['name'],
                gpu_memory_gb=round(gpu_info['memory_gb'], 2) if gpu_info['memory_gb'] else None,
                cuda_available=gpu_info['cuda_available'],
                average_cpu_load=round(average_cpu_load, 2),
                system_platform=platform.system(),
                recommended_mode=recommended_mode,
                performance_score=round(performance_score, 2),
                timestamp=datetime.now()
            )
            
            logger.info(f"System profiling complete: {specs.recommended_mode} performance mode recommended")
            return specs
            
        except Exception as e:
            logger.error(f"Error profiling system: {e}")
            # Return conservative defaults
            return SystemSpecs(
                cpu_cores=2, cpu_frequency_mhz=2000, total_ram_gb=4.0, 
                available_ram_gb=2.0, gpu_available=False, gpu_name=None,
                gpu_memory_gb=None, cuda_available=False, average_cpu_load=50.0,
                system_platform=platform.system(), recommended_mode="low",
                performance_score=30.0, timestamp=datetime.now()
            )
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information using multiple methods (Python 3.12+ compatible)
        """
        gpu_info = {
            'available': False,
            'name': None,
            'memory_gb': None,
            'cuda_available': False,
            'utilization': None
        }
        
        # Method 1: Try nvidia-ml-py3 (most reliable)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_info['available'] = True
            gpu_info['name'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info['memory_gb'] = mem_info.total / (1024**3)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info['utilization'] = util.gpu
            
            gpu_info['cuda_available'] = self._check_cuda_availability()
            
            logger.info(f"GPU detected via pynvml: {gpu_info['name']}")
            return gpu_info
            
        except ImportError:
            logger.debug("pynvml not available, trying nvidia-smi")
        except Exception as e:
            logger.debug(f"pynvml failed: {e}")
        
        # Method 2: Try nvidia-smi command
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 3:
                        gpu_info['available'] = True
                        gpu_info['name'] = parts[0].strip()
                        gpu_info['memory_gb'] = float(parts[1]) / 1024  # MB to GB
                        gpu_info['utilization'] = float(parts[2])
                        gpu_info['cuda_available'] = True  # If nvidia-smi works, CUDA is available
                        
                        logger.info(f"GPU detected via nvidia-smi: {gpu_info['name']}")
                        return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.debug(f"nvidia-smi failed: {e}")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        
        # Method 3: Check for Intel integrated graphics or other GPUs
        try:
            if platform.system() == "Windows":
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 
                    'get', 'name,AdapterRAM', '/format:csv'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip() and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 3 and parts[2].strip():
                                gpu_info['available'] = True
                                gpu_info['name'] = parts[2].strip()
                                if parts[1].strip() and parts[1].strip().isdigit():
                                    gpu_info['memory_gb'] = int(parts[1]) / (1024**3)
                                break
                                
        except Exception as e:
            logger.debug(f"Windows GPU detection failed: {e}")
        
        if not gpu_info['available']:
            logger.info("No GPU detected")
        
        return gpu_info
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
            
        # Try nvcc (CUDA compiler)
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
            
        return False
    
    def _calculate_performance_score(self, cpu_cores: int, cpu_freq: float, 
                                   total_ram: float, available_ram: float,
                                   gpu_available: bool, gpu_memory: Optional[float],
                                   cpu_load: float, cuda_available: bool) -> float:
        """
        Calculate overall performance score (0-100)
        Higher score = better performance = can handle high-performance mode
        """
        score = 0
        
        # CPU Score (40% weight)
        cpu_score = min(cpu_cores * 5, 20)  # Up to 20 points for CPU cores
        cpu_score += min(cpu_freq / 1000 * 10, 20)  # Up to 20 points for frequency
        cpu_score *= (1 - cpu_load / 100 * 0.5)  # Reduce if high current load
        
        # Memory Score (30% weight)
        memory_score = min(total_ram * 5, 15)  # Up to 15 points for total RAM
        memory_score += min(available_ram * 10, 15)  # Up to 15 points for available RAM
        
        # GPU Score (30% weight)
        gpu_score = 0
        if gpu_available:
            gpu_score += 15  # Base points for having GPU
            if gpu_memory:
                gpu_score += min(gpu_memory * 2, 10)  # Up to 10 points for VRAM
            if cuda_available:
                gpu_score += 5  # Bonus for CUDA support
        
        total_score = cpu_score + memory_score + gpu_score
        return min(total_score, 100)
    
    def monitor_runtime_performance(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """
        Monitor system performance during actual operation
        """
        logger.info(f"Starting {duration_seconds}s performance monitoring...")
        
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration_seconds:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0.5),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            # Add GPU utilization if available
            gpu_util = self._get_gpu_utilization()
            if gpu_util:
                sample.update(gpu_util)
            
            samples.append(sample)
            time.sleep(0.5)
        
        # Analyze performance
        avg_cpu = sum(s['cpu_percent'] for s in samples) / len(samples)
        avg_memory = sum(s['memory_percent'] for s in samples) / len(samples)
        min_available_memory = min(s['memory_available_gb'] for s in samples)
        
        # Performance assessment
        performance_good = (
            avg_cpu < 80 and 
            avg_memory < 85 and 
            min_available_memory > 1.0
        )
        
        result = {
            'duration_seconds': duration_seconds,
            'samples_collected': len(samples),
            'average_cpu_percent': round(avg_cpu, 2),
            'average_memory_percent': round(avg_memory, 2),
            'min_available_memory_gb': round(min_available_memory, 2),
            'performance_sustainable': performance_good,
            'recommended_mode': 'high' if performance_good else 'low',
            'samples': samples[-5:]  # Last 5 samples for debugging
        }
        
        logger.info(f"Performance monitoring complete: {result['recommended_mode']} mode recommended")
        return result
    
    def _get_gpu_utilization(self) -> Optional[Dict[str, float]]:
        """Get current GPU utilization"""
        try:
            # Try pynvml first
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'gpu_utilization': util.gpu,
                'gpu_memory_used_gb': mem_info.used / (1024**3),
                'gpu_memory_free_gb': mem_info.free / (1024**3)
            }
        except:
            pass
            
        # Try nvidia-smi
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    return {
                        'gpu_utilization': float(parts[0]),
                        'gpu_memory_used_gb': float(parts[1]) / 1024,
                        'gpu_memory_free_gb': float(parts[2]) / 1024
                    }
        except:
            pass
            
        return None
    
    def get_performance_recommendation(self, include_runtime_test: bool = False) -> Dict[str, Any]:
        """Get comprehensive performance recommendation"""
        # Get static system specs
        specs = self.get_system_specs()
        
        result = {
            'system_specs': {
                'cpu_cores': specs.cpu_cores,
                'cpu_frequency_mhz': specs.cpu_frequency_mhz,
                'total_ram_gb': specs.total_ram_gb,
                'available_ram_gb': specs.available_ram_gb,
                'gpu_available': specs.gpu_available,
                'gpu_name': specs.gpu_name,
                'gpu_memory_gb': specs.gpu_memory_gb,
                'cuda_available': specs.cuda_available,
                'system_platform': specs.system_platform,
                'average_cpu_load': specs.average_cpu_load
            },
            'performance_score': specs.performance_score,
            'static_recommendation': specs.recommended_mode,
            'timestamp': specs.timestamp.isoformat()
        }
        
        # Optional runtime performance test
        if include_runtime_test:
            runtime_perf = self.monitor_runtime_performance(5)  # Shorter test
            result['runtime_performance'] = runtime_perf
            
            # Final recommendation combines both
            final_mode = 'high' if (
                specs.recommended_mode == 'high' and 
                runtime_perf['performance_sustainable']
            ) else 'low'
            
            result['final_recommendation'] = final_mode
        else:
            result['final_recommendation'] = specs.recommended_mode
        
        # Add performance thresholds and reasoning
        result['thresholds'] = {
            'high_performance_score_min': self.high_performance_threshold,
            'min_ram_gb': 6.0,
            'min_cpu_cores': 4,
            'cuda_preferred': True
        }
        
        result['reasoning'] = self._get_recommendation_reasoning(specs)
        
        return result
    
    def _get_recommendation_reasoning(self, specs: SystemSpecs) -> Dict[str, Any]:
        """Provide reasoning for the recommendation"""
        reasons = {
            'mode': specs.recommended_mode,
            'primary_factors': [],
            'limiting_factors': [],
            'suggestions': []
        }
        
        if specs.performance_score >= self.high_performance_threshold:
            reasons['primary_factors'].append(f"High performance score: {specs.performance_score}/100")
        else:
            reasons['limiting_factors'].append(f"Low performance score: {specs.performance_score}/100")
        
        if specs.cpu_cores >= 4:
            reasons['primary_factors'].append(f"Sufficient CPU cores: {specs.cpu_cores}")
        else:
            reasons['limiting_factors'].append(f"Limited CPU cores: {specs.cpu_cores}")
        
        if specs.available_ram_gb >= 6:
            reasons['primary_factors'].append(f"Sufficient RAM: {specs.available_ram_gb}GB available")
        else:
            reasons['limiting_factors'].append(f"Limited RAM: {specs.available_ram_gb}GB available")
        
        if specs.gpu_available and specs.cuda_available:
            reasons['primary_factors'].append(f"GPU with CUDA: {specs.gpu_name}")
        elif specs.gpu_available:
            reasons['primary_factors'].append(f"GPU available: {specs.gpu_name}")
        else:
            reasons['limiting_factors'].append("No GPU detected")
        
        # Suggestions
        if specs.recommended_mode == 'low':
            reasons['suggestions'].append("Consider upgrading hardware for real-time detection")
            reasons['suggestions'].append("Close other applications to free up resources")
            reasons['suggestions'].append("Use on-demand detection to reduce system load")
        else:
            reasons['suggestions'].append("System capable of real-time detection")
            reasons['suggestions'].append("Monitor system performance during operation")
        
        return reasons

# Global profiler instance
system_profiler = SystemProfiler()