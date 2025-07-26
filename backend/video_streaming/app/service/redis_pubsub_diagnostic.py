# redis_pubsub_diagnostic.py - Debug tool for Redis pubsub issues
import asyncio
import redis
import redis.asyncio as async_redis
import pickle
import time
import logging
from typing import Dict, Any, List
import threading
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisPubSubDiagnostic:
    """Comprehensive diagnostic tool for Redis pubsub communication issues"""
    
    def __init__(self, redis_host='redis', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Create both sync and async clients
        self.sync_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=0, 
            decode_responses=False
        )
        self.async_client = None
        
        # Diagnostic results
        self.test_results = {}
        
    async def initialize_async_client(self):
        """Initialize async Redis client"""
        self.async_client = async_redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=0,
            decode_responses=False,
            socket_keepalive=True,
            health_check_interval=30
        )
        await self.async_client.ping()
        
    async def run_comprehensive_diagnosis(self, camera_id: int = 4) -> Dict[str, Any]:
        """Run comprehensive diagnosis of Redis pubsub issues"""
        logger.info(f"🔍 Starting comprehensive Redis pubsub diagnosis for camera {camera_id}")
        
        await self.initialize_async_client()
        
        results = {
            'camera_id': camera_id,
            'timestamp': time.time(),
            'tests': {}
        }
        
        # Test 1: Basic Redis connectivity
        results['tests']['redis_connectivity'] = await self._test_redis_connectivity()
        
        # Test 2: Channel inspection
        results['tests']['channel_inspection'] = await self._inspect_channels(camera_id)
        
        # Test 3: Pubsub message flow test
        results['tests']['message_flow'] = await self._test_message_flow(camera_id)
        
        # Test 4: Detection queue inspection
        results['tests']['detection_queue'] = await self._inspect_detection_queue()
        
        # Test 5: Processed frame cache inspection
        results['tests']['processed_frames'] = await self._inspect_processed_frames(camera_id)
        
        # Test 6: Cross-service communication test
        results['tests']['cross_service'] = await self._test_cross_service_communication(camera_id)
        
        # Test 7: Redis key analysis
        results['tests']['redis_keys'] = await self._analyze_redis_keys(camera_id)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['tests'])
        
        return results
    
    async def _test_redis_connectivity(self) -> Dict[str, Any]:
        """Test basic Redis connectivity"""
        try:
            # Test sync client
            sync_ping = self.sync_client.ping()
            
            # Test async client
            async_ping = await self.async_client.ping()
            
            # Test connection info
            info = self.sync_client.info()
            
            return {
                'status': 'success',
                'sync_ping': sync_ping,
                'async_ping': async_ping,
                'redis_version': info.get('redis_version', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown')
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _inspect_channels(self, camera_id: int) -> Dict[str, Any]:
        """Inspect pubsub channels and subscribers"""
        try:
            detection_channel = f"detection_results:{camera_id}"
            
            # Get subscriber count using sync client
            subscribers_result = self.sync_client.pubsub_numsub(detection_channel)
            subscriber_count = subscribers_result[0][1] if subscribers_result and len(subscribers_result[0]) > 1 else 0
            
            # Get active channels
            active_channels = self.sync_client.pubsub_channels()
            
            # Get pattern subscribers
            pattern_subscribers = self.sync_client.pubsub_numpat()
            
            return {
                'status': 'success',
                'detection_channel': detection_channel,
                'subscriber_count': subscriber_count,
                'active_channels': [ch.decode() if isinstance(ch, bytes) else ch for ch in active_channels],
                'pattern_subscribers': pattern_subscribers,
                'channel_exists': detection_channel.encode() in active_channels or detection_channel in active_channels
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _test_message_flow(self, camera_id: int) -> Dict[str, Any]:
        """Test end-to-end message flow"""
        try:
            detection_channel = f"detection_results:{camera_id}"
            
            # Create test subscriber
            test_received_messages = []
            subscriber_task = None
            
            try:
                # Create pubsub connection
                pubsub = self.async_client.pubsub()
                await pubsub.subscribe(detection_channel)
                
                # Wait for subscription confirmation
                subscription_confirmed = False
                for _ in range(10):  # 5 second timeout
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=False),
                        timeout=0.5
                    )
                    if message and message['type'] == 'subscribe':
                        subscription_confirmed = True
                        break
                
                if not subscription_confirmed:
                    raise Exception("Subscription confirmation timeout")
                
                # Start message listener
                async def message_listener():
                    try:
                        for _ in range(20):  # Listen for up to 10 seconds
                            message = await asyncio.wait_for(
                                pubsub.get_message(ignore_subscribe_messages=True),
                                timeout=0.5
                            )
                            if message and message['type'] == 'message':
                                try:
                                    data = pickle.loads(message['data'])
                                    test_received_messages.append(data)
                                except:
                                    test_received_messages.append({'raw_message': message})
                    except asyncio.TimeoutError:
                        pass
                
                subscriber_task = asyncio.create_task(message_listener())
                
                # Give subscriber time to start
                await asyncio.sleep(0.5)
                
                # Publish test messages
                test_messages = []
                for i in range(3):
                    test_data = {
                        'camera_id': camera_id,
                        'test_message': True,
                        'message_id': i,
                        'timestamp': time.time(),
                        'detected_target': i % 2 == 0,
                        'session_id': 'diagnostic_test'
                    }
                    
                    serialized = pickle.dumps(test_data)
                    published_count = await self.async_client.publish(detection_channel, serialized)
                    
                    test_messages.append({
                        'message_id': i,
                        'published_to_subscribers': published_count,
                        'data': test_data
                    })
                    
                    await asyncio.sleep(0.5)
                
                # Wait for messages to be received
                await asyncio.sleep(2.0)
                
                # Cancel subscriber
                if subscriber_task:
                    subscriber_task.cancel()
                    try:
                        await subscriber_task
                    except asyncio.CancelledError:
                        pass
                
                # Cleanup pubsub
                await pubsub.unsubscribe()
                await pubsub.aclose()
                
                return {
                    'status': 'success',
                    'subscription_confirmed': subscription_confirmed,
                    'messages_published': len(test_messages),
                    'messages_received': len(test_received_messages),
                    'published_messages': test_messages,
                    'received_messages': test_received_messages,
                    'message_flow_working': len(test_received_messages) > 0
                }
                
            except Exception as e:
                # Cleanup on error
                if subscriber_task:
                    subscriber_task.cancel()
                return {
                    'status': 'error',
                    'error': str(e),
                    'messages_received': len(test_received_messages)
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _inspect_detection_queue(self) -> Dict[str, Any]:
        """Inspect the detection queue"""
        try:
            queue_name = 'detection_queue'
            
            # Get queue length
            queue_length = self.sync_client.llen(queue_name)
            
            # Get a few items from the queue (without removing them)
            queue_items = []
            if queue_length > 0:
                # Get up to 3 items
                items = self.sync_client.lrange(queue_name, 0, 2)
                for item in items:
                    try:
                        deserialized = pickle.loads(item)
                        queue_items.append({
                            'camera_id': deserialized.get('camera_id'),
                            'target_label': deserialized.get('target_label'),
                            'timestamp': deserialized.get('timestamp'),
                            'session_id': deserialized.get('session_id'),
                            'frame_size': len(deserialized.get('frame_data', b''))
                        })
                    except Exception as e:
                        queue_items.append({'error': f'Failed to deserialize: {e}'})
            
            return {
                'status': 'success',
                'queue_length': queue_length,
                'queue_items': queue_items,
                'queue_active': queue_length > 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _inspect_processed_frames(self, camera_id: int) -> Dict[str, Any]:
        """Inspect processed frame cache"""
        try:
            # Look for processed frame keys
            pattern = f"processed_frame:{camera_id}:*"
            keys = self.sync_client.keys(pattern)
            
            processed_frames = []
            for key in keys[:5]:  # Check up to 5 keys
                try:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    ttl = self.sync_client.ttl(key)
                    frame_data = self.sync_client.get(key)
                    
                    processed_frames.append({
                        'key': key_str,
                        'ttl_seconds': ttl,
                        'frame_size': len(frame_data) if frame_data else 0,
                        'exists': frame_data is not None
                    })
                except Exception as e:
                    processed_frames.append({
                        'key': key.decode() if isinstance(key, bytes) else str(key),
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'processed_frame_keys_found': len(keys),
                'processed_frames': processed_frames,
                'frames_available': len(keys) > 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _test_cross_service_communication(self, camera_id: int) -> Dict[str, Any]:
        """Test communication between detection service and video service"""
        try:
            detection_channel = f"detection_results:{camera_id}"
            
            # 1. Check if detection service is publishing (simulate a detection request)
            test_frame_request = {
                'camera_id': camera_id,
                'frame_data': b'test_frame_data',  # Small test data
                'target_label': 'test_target',
                'timestamp': time.time(),
                'session_id': 'diagnostic_test',
                'priority': 1
            }
            
            # Send to detection queue
            serialized_request = pickle.dumps(test_frame_request)
            queue_length = self.sync_client.lpush('detection_queue', serialized_request)
            
            # 2. Monitor for results for a short time
            pubsub = self.async_client.pubsub()
            await pubsub.subscribe(detection_channel)
            
            # Wait for subscription
            subscription_ok = False
            for _ in range(10):
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=False),
                    timeout=0.5
                )
                if message and message['type'] == 'subscribe':
                    subscription_ok = True
                    break
            
            detection_results = []
            if subscription_ok:
                # Listen for detection results for 10 seconds
                try:
                    for _ in range(20):
                        message = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True),
                            timeout=0.5
                        )
                        if message and message['type'] == 'message':
                            try:
                                result = pickle.loads(message['data'])
                                if result.get('session_id') == 'diagnostic_test':
                                    detection_results.append(result)
                                    break  # Found our test result
                            except:
                                pass
                except asyncio.TimeoutError:
                    pass
            
            await pubsub.unsubscribe()
            await pubsub.aclose()
            
            return {
                'status': 'success',
                'test_request_queued': True,
                'queue_length_after_request': queue_length,
                'subscription_established': subscription_ok,
                'detection_results_received': len(detection_results),
                'cross_service_working': len(detection_results) > 0,
                'detection_results': detection_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _analyze_redis_keys(self, camera_id: int) -> Dict[str, Any]:
        """Analyze Redis keys related to detection"""
        try:
            # Get all keys related to our camera
            patterns = [
                f"detection_result:{camera_id}:*",
                f"processed_frame:{camera_id}:*",
                "detection_queue"
            ]
            
            key_analysis = {}
            for pattern in patterns:
                keys = self.sync_client.keys(pattern)
                key_analysis[pattern] = {
                    'count': len(keys),
                    'keys': [k.decode() if isinstance(k, bytes) else k for k in keys[:10]]  # First 10 keys
                }
            
            # Check Redis memory usage
            info = self.sync_client.info('memory')
            
            return {
                'status': 'success',
                'key_analysis': key_analysis,
                'memory_info': {
                    'used_memory_human': info.get('used_memory_human', 'unknown'),
                    'used_memory_peak_human': info.get('used_memory_peak_human', 'unknown'),
                    'total_system_memory_human': info.get('total_system_memory_human', 'unknown')
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check Redis connectivity
        if test_results.get('redis_connectivity', {}).get('status') != 'success':
            recommendations.append("❌ CRITICAL: Fix Redis connectivity issues first")
        
        # Check channel subscription
        channel_info = test_results.get('channel_inspection', {})
        if channel_info.get('subscriber_count', 0) == 0:
            recommendations.append("⚠️ No subscribers found on detection channel - video service may not be listening")
        
        # Check message flow
        message_flow = test_results.get('message_flow', {})
        if not message_flow.get('message_flow_working', False):
            recommendations.append("❌ CRITICAL: Message flow test failed - pubsub communication broken")
            if not message_flow.get('subscription_confirmed', False):
                recommendations.append("   └── Subscription confirmation failed - check async Redis client")
        
        # Check detection queue
        queue_info = test_results.get('detection_queue', {})
        if queue_info.get('queue_length', 0) > 10:
            recommendations.append("⚠️ Detection queue is backing up - detection service may be slow")
        elif queue_info.get('queue_length', 0) == 0:
            recommendations.append("ℹ️ Detection queue is empty - may indicate no frames being sent for detection")
        
        # Check processed frames
        frames_info = test_results.get('processed_frames', {})
        if not frames_info.get('frames_available', False):
            recommendations.append("❌ No processed frames found in cache - detection service may not be storing results")
        
        # Check cross-service communication
        cross_service = test_results.get('cross_service', {})
        if not cross_service.get('cross_service_working', False):
            recommendations.append("❌ CRITICAL: Cross-service communication test failed")
            if cross_service.get('test_request_queued', False) and cross_service.get('detection_results_received', 0) == 0:
                recommendations.append("   └── Detection service is not processing requests or not publishing results")
        
        if not recommendations:
            recommendations.append("✅ All tests passed - pubsub communication appears to be working")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.async_client:
            await self.async_client.aclose()
        if self.sync_client:
            self.sync_client.close()

# Standalone diagnostic function
async def diagnose_redis_pubsub_issues(camera_id: int = 4, redis_host: str = 'redis', redis_port: int = 6379):
    """Run comprehensive Redis pubsub diagnosis"""
    diagnostic = RedisPubSubDiagnostic(redis_host, redis_port)
    
    try:
        results = await diagnostic.run_comprehensive_diagnosis(camera_id)
        
        # Print formatted results
        print("=" * 80)
        print(f"🔍 REDIS PUBSUB DIAGNOSTIC REPORT - Camera {camera_id}")
        print("=" * 80)
        
        for test_name, test_result in results['tests'].items():
            print(f"\n📊 {test_name.upper()}")
            print("-" * 40)
            
            if test_result.get('status') == 'success':
                print("✅ Status: PASSED")
                for key, value in test_result.items():
                    if key != 'status':
                        print(f"   {key}: {value}")
            else:
                print("❌ Status: FAILED")
                print(f"   Error: {test_result.get('error', 'Unknown error')}")
        
        print(f"\n🎯 RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "=" * 80)
        
        return results
        
    finally:
        await diagnostic.cleanup()

# CLI entry point
if __name__ == "__main__":
    import sys
    
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    redis_host = sys.argv[2] if len(sys.argv) > 2 else 'redis'
    redis_port = int(sys.argv[3]) if len(sys.argv) > 3 else 6379
    
    asyncio.run(diagnose_redis_pubsub_issues(camera_id, redis_host, redis_port))