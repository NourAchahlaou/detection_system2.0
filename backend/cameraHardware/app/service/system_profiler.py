# system_profiler.py - Updated with realistic thresholds for AI detection system
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
    meets_minimum_requirements: bool
    performance_tier: str  # "low", "medium", "high"

class SystemProfiler:
    """
    System profiler that determines optimal performance mode for AI detection system
    Updated with realistic thresholds for YOLO + Redis + Video Streaming
    """
    
    def __init__(self):
        self.performance_history = []
        self.monitoring_duration = 10
        
        # UPDATED: Realistic thresholds for AI detection system
        self.minimum_requirements = {
            'cpu_cores': 6,           # Increased from 4
            'total_ram_gb': 12.0,     # Increased from 6
            'available_ram_gb': 8.0,  # Increased for safety margin
            'cpu_frequency_mhz': 2500, # Minimum for real-time processing
            'gpu_memory_gb': 6.0      # Minimum VRAM for YOLO + training
        }
        
        self.performance_thresholds = {
            'high_performance_score': 80,    # Increased threshold
            'medium_performance_score': 60,  # New medium tier
            'low_performance_score': 40      # Basic functionality
        }
        
    def get_system_specs(self) -> SystemSpecs:
        """Get comprehensive system specifications with updated scoring"""
        try:
            # CPU Information
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current if cpu_freq else 0
            
            # Memory Information
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            # CPU Load (sample over short period for accuracy)
            cpu_loads = []
            for _ in range(5):
                cpu_loads.append(psutil.cpu_percent(interval=0.5))
            average_cpu_load = sum(cpu_loads) / len(cpu_loads)
            
            # GPU Information
            gpu_info = self._get_gpu_info()
            
            # Check minimum requirements
            meets_minimum = self._check_minimum_requirements(
                cpu_cores, cpu_frequency, total_ram_gb, available_ram_gb,
                gpu_info['memory_gb'], gpu_info['cuda_available']
            )
            
            # Calculate performance score with updated algorithm
            performance_score = self._calculate_performance_score_v2(
                cpu_cores, cpu_frequency, total_ram_gb, available_ram_gb,
                gpu_info['available'], gpu_info['memory_gb'], average_cpu_load, 
                gpu_info['cuda_available']
            )
            
            # Determine performance tier and mode
            performance_tier, recommended_mode = self._determine_performance_tier(
                performance_score, meets_minimum
            )
            
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
                timestamp=datetime.now(),
                meets_minimum_requirements=meets_minimum,
                performance_tier=performance_tier
            )
            
            logger.info(f"System profiling complete: {specs.performance_tier} tier, "
                       f"{specs.recommended_mode} mode, meets minimum: {meets_minimum}")
            return specs
            
        except Exception as e:
            logger.error(f"Error profiling system: {e}")
            # Return conservative defaults
            return SystemSpecs(
                cpu_cores=2, cpu_frequency_mhz=2000, total_ram_gb=4.0, 
                available_ram_gb=2.0, gpu_available=False, gpu_name=None,
                gpu_memory_gb=None, cuda_available=False, average_cpu_load=50.0,
                system_platform=platform.system(), recommended_mode="basic",
                performance_score=20.0, timestamp=datetime.now(),
                meets_minimum_requirements=False, performance_tier="low"
            )
    
    def _check_minimum_requirements(self, cpu_cores: int, cpu_freq: float, 
                                   total_ram: float, available_ram: float,
                                   gpu_memory: Optional[float], cuda_available: bool) -> bool:
        """Check if system meets minimum requirements for advanced mode"""
        
        requirements_met = {
            'cpu_cores': cpu_cores >= self.minimum_requirements['cpu_cores'],
            'total_ram': total_ram >= self.minimum_requirements['total_ram_gb'],
            'available_ram': available_ram >= self.minimum_requirements['available_ram_gb'],
            'cpu_frequency': cpu_freq >= self.minimum_requirements['cpu_frequency_mhz'],
            'gpu_memory': (gpu_memory or 0) >= self.minimum_requirements['gpu_memory_gb'] if cuda_available else True
        }
        
        # Log which requirements are not met
        failed_requirements = [req for req, met in requirements_met.items() if not met]
        if failed_requirements:
            logger.warning(f"System does not meet minimum requirements: {failed_requirements}")
        
        # Must meet all critical requirements (CPU, RAM)
        critical_requirements = ['cpu_cores', 'total_ram', 'available_ram']
        return all(requirements_met[req] for req in critical_requirements)
    
    def _calculate_performance_score_v2(self, cpu_cores: int, cpu_freq: float, 
                                       total_ram: float, available_ram: float,
                                       gpu_available: bool, gpu_memory: Optional[float],
                                       cpu_load: float, cuda_available: bool) -> float:
        """
        Updated performance scoring algorithm for AI detection system
        Score 0-100, higher = better performance
        """
        score = 0
        
        # CPU Score (35% weight) - More emphasis on cores for parallel processing
        cpu_core_score = min(cpu_cores * 5, 30)  # Up to 30 points (6 cores = 30)
        cpu_freq_score = min((cpu_freq - 2000) / 1000 * 15, 15)  # Up to 15 points
        cpu_load_penalty = (cpu_load / 100) * 10  # Penalty for high current load
        cpu_score = max(0, cpu_core_score + cpu_freq_score - cpu_load_penalty)
        
        # Memory Score (35% weight) - Critical for AI models
        ram_score = min(total_ram * 2, 20)  # Up to 20 points (10GB = 20 points)
        available_ram_score = min(available_ram * 2, 15)  # Up to 15 points
        memory_score = ram_score + available_ram_score
        
        # GPU Score (30% weight) - Important for YOLO inference
        gpu_score = 0
        if gpu_available and cuda_available:
            gpu_score += 15  # Base points for CUDA GPU
            if gpu_memory:
                # 6GB VRAM = 15 points, scales up
                gpu_score += min(gpu_memory * 2.5, 15)
        elif gpu_available:
            gpu_score += 5  # Basic GPU without CUDA
        
        total_score = cpu_score + memory_score + gpu_score
        
        # Apply system-specific bonuses/penalties
        if platform.system() == "Linux":
            total_score += 2  # Linux typically performs better for AI workloads
        
        return min(total_score, 100)
    
    def _determine_performance_tier(self, score: float, meets_minimum: bool) -> tuple[str, str]:
        """Determine performance tier and recommended mode"""
        
        if not meets_minimum:
            return "low", "basic"
        
        if score >= self.performance_thresholds['high_performance_score']:
            return "high", "advanced"
        elif score >= self.performance_thresholds['medium_performance_score']:
            return "medium", "advanced"
        else:
            return "low", "basic"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information with better VRAM detection"""
        gpu_info = {
            'available': False,
            'name': None,
            'memory_gb': None,
            'cuda_available': False,
            'utilization': None
        }
        
        # Method 1: Try nvidia-ml-py3 (most reliable for NVIDIA)
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
            
            logger.info(f"GPU detected via pynvml: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
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
                        gpu_info['cuda_available'] = True
                        
                        logger.info(f"GPU detected via nvidia-smi: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
                        return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.debug(f"nvidia-smi failed: {e}")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        
        if not gpu_info['available']:
            logger.info("No compatible GPU detected for AI acceleration")
        
        return gpu_info
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available and functional"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
            
        # Try PyTorch CUDA check if available
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
            
        return False
    
    def get_performance_recommendation(self, include_runtime_test: bool = False) -> Dict[str, Any]:
        """Get comprehensive performance recommendation with detailed analysis"""
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
            'performance_tier': specs.performance_tier,
            'meets_minimum_requirements': specs.meets_minimum_requirements,
            'static_recommendation': specs.recommended_mode,
            'timestamp': specs.timestamp.isoformat()
        }
        
        # Optional runtime performance test
        if include_runtime_test:
            runtime_perf = self.monitor_runtime_performance(5)
            result['runtime_performance'] = runtime_perf
            
            # Final recommendation combines both
            final_mode = 'advanced' if (
                specs.recommended_mode == 'advanced' and 
                runtime_perf['performance_sustainable']
            ) else 'basic'
            
            result['final_recommendation'] = final_mode
        else:
            result['final_recommendation'] = specs.recommended_mode
        
        # Add detailed thresholds and reasoning
        result['minimum_requirements'] = self.minimum_requirements
        result['performance_thresholds'] = self.performance_thresholds
        result['reasoning'] = self._get_recommendation_reasoning(specs)
        
        return result
    
    def _get_recommendation_reasoning(self, specs: SystemSpecs) -> Dict[str, Any]:
        """Enhanced reasoning for the recommendation"""
        reasons = {
            'mode': specs.recommended_mode,
            'tier': specs.performance_tier,
            'meets_minimum': specs.meets_minimum_requirements,
            'strengths': [],
            'limitations': [],
            'suggestions': [],
            'system_breakdown': {}
        }
        
        # Analyze each component
        if specs.cpu_cores >= self.minimum_requirements['cpu_cores']:
            reasons['strengths'].append(f"Sufficient CPU cores: {specs.cpu_cores}")
        else:
            reasons['limitations'].append(f"Insufficient CPU cores: {specs.cpu_cores} (need {self.minimum_requirements['cpu_cores']})")
        
        if specs.total_ram_gb >= self.minimum_requirements['total_ram_gb']:
            reasons['strengths'].append(f"Adequate RAM: {specs.total_ram_gb}GB")
        else:
            reasons['limitations'].append(f"Insufficient RAM: {specs.total_ram_gb}GB (need {self.minimum_requirements['total_ram_gb']}GB)")
        
        if specs.gpu_available and specs.cuda_available:
            if specs.gpu_memory_gb and specs.gpu_memory_gb >= self.minimum_requirements['gpu_memory_gb']:
                reasons['strengths'].append(f"Powerful GPU with CUDA: {specs.gpu_name} ({specs.gpu_memory_gb}GB)")
            else:
                reasons['limitations'].append(f"GPU VRAM insufficient: {specs.gpu_memory_gb}GB (recommended {self.minimum_requirements['gpu_memory_gb']}GB)")
        elif specs.gpu_available:
            reasons['limitations'].append(f"GPU available but no CUDA support: {specs.gpu_name}")
        else:
            reasons['limitations'].append("No GPU detected - will use CPU-only inference")
        
        # System breakdown
        reasons['system_breakdown'] = {
            'cpu_adequacy': 'adequate' if specs.cpu_cores >= 6 else 'limited',
            'memory_adequacy': 'adequate' if specs.total_ram_gb >= 12 else 'limited',
            'gpu_adequacy': 'adequate' if specs.gpu_available and specs.cuda_available else 'limited',
            'overall_capability': specs.performance_tier
        }
        
        # Suggestions based on limitations
        if specs.recommended_mode == 'basic':
            reasons['suggestions'].extend([
                "Consider upgrading to meet minimum requirements for advanced mode",
                "Use basic detection mode for optimal performance",
                "Consider adding GPU acceleration for better performance"
            ])
            
            if specs.cpu_cores < 6:
                reasons['suggestions'].append("Upgrade to at least 6-core CPU for better performance")
            if specs.total_ram_gb < 12:
                reasons['suggestions'].append("Upgrade RAM to at least 12GB for AI model loading")
            if not specs.cuda_available:
                reasons['suggestions'].append("Add NVIDIA GPU with CUDA for accelerated inference")
        else:
            reasons['suggestions'].extend([
                "System capable of advanced real-time detection",
                "Monitor system performance during heavy usage",
                "Consider disabling background training during peak detection periods"
            ])
        
        return reasons
    
    def monitor_runtime_performance(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Monitor system performance during actual operation"""
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
        
        # Updated performance thresholds for AI workload
        performance_good = (
            avg_cpu < 75 and  # Allow higher CPU usage for AI
            avg_memory < 80 and  # Allow higher memory usage
            min_available_memory > 2.0  # Ensure enough free memory
        )
        
        result = {
            'duration_seconds': duration_seconds,
            'samples_collected': len(samples),
            'average_cpu_percent': round(avg_cpu, 2),
            'average_memory_percent': round(avg_memory, 2),
            'min_available_memory_gb': round(min_available_memory, 2),
            'performance_sustainable': performance_good,
            'recommended_mode': 'advanced' if performance_good else 'basic',
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

# Global profiler instance
system_profiler = SystemProfiler()