import asyncio
import csv
import logging
import sys
import random
import socket
import struct
import time
import signal
import aiodns
import fastapi
import uvicorn

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pythonping import ping

DEBUG_MODE = True

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('host_monitor.log'),
        logging.StreamHandler()
    ]
)
sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger(__name__)

class Host(BaseModel):
    id: str
    address: str
    dns_info: str = None
    is_monitoring: bool = True
    is_up: bool = False
    ping_history: list = []
    latency_history: list = []
    last_checked: datetime = None
    response_log: list = []
    notification_mode: str = 'disabled'
    last_status_change: datetime = None
    
    class Config:
        @staticmethod
        def default_list():
            return []

        fields = {
            'ping_history': {'default_factory': default_list},
            'latency_history': {'default_factory': default_list},
            'response_log': {'default_factory': default_list}
        }

    def maintain_history_size(self, max_size=100):
        if len(self.ping_history) > max_size:
            self.ping_history = self.ping_history[-max_size:]
        if len(self.latency_history) > max_size:
            self.latency_history = self.latency_history[-max_size:]
        if len(self.response_log) > max_size:
            self.response_log = self.response_log[-max_size:]

class HostMonitor:
    def __init__(self, hosts_file='hosts.csv'):
        self.hosts = {}
        self.load_hosts(hosts_file)
        self.websocket_clients = set()
        self.executor = ThreadPoolExecutor(max_workers=100)

    async def _calculate_host_data(self):
        """Calculate fresh host data without caching"""
        host_data = []
        current_time = datetime.now()   
        
        for host in self.hosts.values():
            valid_latencies = [lat for lat in host.latency_history if lat is not None]
            ping_history = host.ping_history

            needs_notification = False
            if host.last_status_change:
                time_since_change = (current_time - host.last_status_change).total_seconds()
                if host.notification_mode == 'notify_up' and host.is_up:
                    needs_notification = True
                elif host.notification_mode == 'notify_down' and not host.is_up:
                    needs_notification = True

            data = {
                'id': host.id,
                'address': host.address,
                'dns_info': host.dns_info,
                'is_up': host.is_up,
                'is_monitoring': host.is_monitoring,
                'ping_success_rate': (sum(ping_history) / len(ping_history) * 100) if ping_history else 0,
                'avg_latency': sum(valid_latencies) / len(valid_latencies) if valid_latencies else None,
                'last_checked': host.last_checked.isoformat() if host.last_checked else None,
                'notification_mode': host.notification_mode,
                'needs_notification': needs_notification,
            }
            host_data.append(data)
        return host_data

    async def notify_clients(self):
        if not self.websocket_clients:
            return

        host_data = await self._calculate_host_data()
        await self._send_updates_to_clients(host_data)

    async def _send_updates_to_clients(self, host_data):
        """Send updates to all connected websocket clients"""
        for websocket in self.websocket_clients.copy():
            try:
                await websocket.send_json(host_data)
            except Exception as e:
                logger.error(f"Failed to send data to websocket: {e}")
                self.websocket_clients.remove(websocket)

    async def ping_host(self, host):
        """Ping host with detailed debug logging"""
        logger.info(f"Pinging {host.address}")
        try:
            loop = asyncio.get_event_loop()
            logger.debug(f"Sending ICMP request to {host.address}")
            
            response = await loop.run_in_executor(
                None, 
                lambda: ping(host.address, count=1, timeout=2)
            )
            
            success = response.success()
            latency = response.rtt_avg_ms if success else None
            logger.debug(f"Received response from {host.address}: success={success}, latency={latency}ms")
                    
            # Update status
            previous_status = host.is_up
            host.is_up = success
            host.last_checked = datetime.now()
            
            # Only notify if status changed
            if previous_status != host.is_up:
                host.last_status_change = datetime.now()
                await self.notify_clients()

            # Update histories
            host.ping_history.append(1 if success else 0)
            host.latency_history.append(latency)
            host.response_log.append({
                "timestamp": host.last_checked.isoformat(),
                "success": success,
                "latency": latency
            })
            host.maintain_history_size()

        except Exception as e:
            logger.error(f"Failed to ping {host.address}: {e}")
            if host.is_up:  # Only notify if status changes
                await self.notify_clients()
            host.is_up = False
            host.ping_history.append(0)
            host.latency_history.append(None)
            host.response_log.append({
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            host.maintain_history_size()

    
    async def monitor_hosts(self, shutdown_event: asyncio.Event):
        """Continuously check for hosts that need monitoring and start their loops."""
        while not shutdown_event.is_set():
            try:
                # Create a copy of hosts to avoid RuntimeError during iteration
                hosts = list(self.hosts.values())
                for host in hosts:
                    if host.is_monitoring and not getattr(host, 'monitor_task', None):
                        host.monitor_task = asyncio.create_task(
                            self.host_monitoring_loop(host, shutdown_event)
                        )
                        logger.debug(f"Started monitoring for host: {host.address}")
                await asyncio.sleep(1)  # Reduced sleep time for quicker response
            except Exception as e:
                logger.error(f"Error in monitor_hosts loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def host_monitoring_loop(self, host, shutdown_event):
        """Monitoring loop with error handling"""
        logger.debug(f"Starting monitoring loop for {host.address}")
        while not shutdown_event.is_set() and host.is_monitoring:
            try:
                await self.ping_host(host)
                await self.notify_clients()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error monitoring {host.address}: {e}")
                await asyncio.sleep(5)  # Wait before retrying after an error
        logger.debug(f"Stopped monitoring {host.address}")


    async def notify_loop(self, shutdown_event: asyncio.Event):
        while not shutdown_event.is_set():
            await self.notify_clients()
            await asyncio.sleep(2)

    async def resolve_dns(self, host):
        """Resolve DNS with change detection and debug logging"""
        try:
            previous_dns = host.dns_info
            logger.debug(f"Attempting DNS resolution for {host.address} (previous: {previous_dns})")
            
            if host.address.replace('.', '').isdigit():
                # Offload gethostbyaddr to a thread
                new_dns = await asyncio.to_thread(socket.gethostbyaddr, host.address)
                new_dns = new_dns[0]  # gethostbyaddr returns a tuple
            else:
                # Offload gethostbyname to a thread
                new_dns = await asyncio.to_thread(socket.gethostbyname, host.address)
            
            logger.debug(f"DNS resolution successful: {host.address} -> {new_dns}")
            if new_dns != previous_dns:
                logger.debug(f"DNS change detected: {previous_dns} -> {new_dns}")
                host.dns_info = new_dns
                await self.notify_clients()
        except (socket.herror, socket.gaierror) as e:
            logger.debug(f"DNS resolution failed for {host.address}: {str(e)}")
            if host.dns_info is not None:
                host.dns_info = None
                await self.notify_clients()
        except Exception as e:
            logger.error(f"DNS error: {str(e)}")
            host.dns_info = None

    async def resolve_dns_for_all_hosts(self, shutdown_event: asyncio.Event):
        dns_tasks = set()  # Track running tasks

        while not shutdown_event.is_set():
            new_tasks = {asyncio.create_task(self.resolve_dns(host)) for host in self.hosts.values() if host.is_monitoring}
            dns_tasks.update(new_tasks)

            # Remove completed tasks
            done_tasks = {task for task in dns_tasks if task.done()}
            dns_tasks.difference_update(done_tasks)

            await asyncio.sleep(10)

        # Cancel remaining tasks on shutdown
        for task in dns_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _send_updates_to_clients(self, host_data):
        """Send updates with WebSocket debug logging"""
        logger.debug(f"Preparing update for {len(self.websocket_clients)} clients")
        for websocket in self.websocket_clients.copy():
            try:
                client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
                logger.debug(f"Sending update to {client_info}")
                await websocket.send_json(host_data)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                self.websocket_clients.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                self.websocket_clients.remove(websocket)

    def load_hosts(self, filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    host_address = row[0].strip()
                    host_id = host_address.replace('.', '_')
                    self.hosts[host_id] = Host(
                        id=host_id, 
                        address=host_address
                    )
        except FileNotFoundError:
            logger.warning(f"Hosts file {filename} not found. Creating a new one.")
            # Create an empty hosts file
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])  # Write an empty row to create the file
            self.hosts = {}  # Initialize with an empty host list

    def save_hosts(self, filename='hosts.csv'):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for host in self.hosts.values():
                writer.writerow([host.address])

    async def ping_host(self, host):
        logger.info(f"Pinging {host.address}")
        try:
            loop = asyncio.get_event_loop()
            # Run the blocking ping call in a thread pool
            response = await loop.run_in_executor(
                None,  # Use the default executor
                lambda: ping(host.address, count=1, timeout=2)
            )
            success = response.success()
            latency = response.rtt_avg_ms if success else None
            
            # Update host status
            host.is_up = success
            host.last_checked = datetime.now()
            
            # Append to histories
            host.ping_history.append(1 if success else 0)
            host.latency_history.append(latency)
            host.response_log.append({
                "timestamp": host.last_checked.isoformat(),
                "success": success,
                "latency": latency
            })
            
            # Maintain history size
            host.maintain_history_size()
            
            logger.debug(f"Ping results for {host.address}: Success={success}, Latency={latency}ms")
            
        except Exception as e:
            logger.error(f"Failed to ping {host.address}: {e}")
            host.is_up = False
            host.ping_history.append(0)
            host.latency_history.append(None)
            host.response_log.append({
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            host.maintain_history_size()

    async def monitor_hosts(self, shutdown_event: asyncio.Event):
        """Start individual monitoring loops for each host"""
        tasks = []
        for host in self.hosts.values():
            if host.is_monitoring:
                task = asyncio.create_task(self.host_monitoring_loop(host, shutdown_event))
                tasks.append(task)
        await shutdown_event.wait()
        # Cleanup tasks on shutdown
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def host_monitoring_loop(self, host, shutdown_event):
        """Monitoring loop with start/stop logging"""
        logger.debug(f"Starting monitoring loop for {host.address}")
        while not shutdown_event.is_set() and host.is_monitoring:
            await self.ping_host(host)
            await self.notify_clients()
            await asyncio.sleep(1)
        logger.debug(f"Stopped monitoring {host.address}")





app = fastapi.FastAPI()
host_monitor = HostMonitor()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.debug(f"WebSocket connection opened by {client_info}")
    host_monitor.websocket_clients.add(websocket)
    
    try:
        logger.debug(f"Sending initial data to {client_info}")
        await host_monitor.notify_clients()
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received keep-alive from {client_info}: {data}")
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected by {client_info}")
        host_monitor.websocket_clients.remove(websocket)


@app.post("/hosts")
async def add_host(address: str = fastapi.Form(...)):
    host_id = address.replace('.', '_')
    if host_id not in host_monitor.hosts:
        new_host = Host(id=host_id, address=address, is_monitoring=True)
        host_monitor.hosts[host_id] = new_host
        host_monitor.save_hosts()
        # Immediately start monitoring if applicable
        if new_host.is_monitoring:
            new_host.monitor_task = asyncio.create_task(
                host_monitor.host_monitoring_loop(new_host, host_monitor.shutdown_event)
            )
        await host_monitor.notify_clients()
        return new_host
    return None

@app.get("/hosts")
def get_hosts():
    host_data = []
    for host in host_monitor.hosts.values():
        valid_latencies = [lat for lat in host.latency_history if lat is not None]
        ping_history = host.ping_history

        data = {
            'id': host.id,
            'address': host.address,
            'dns_info': host.dns_info,
            'is_up': host.is_up,
            'is_monitoring': host.is_monitoring,
            'ping_success_rate': (sum(ping_history) / len(ping_history) * 100) if ping_history else 0,
            'avg_latency': sum(valid_latencies) / len(valid_latencies) if valid_latencies else None,
            'last_checked': host.last_checked.isoformat() if host.last_checked else None
        }
        host_data.append(data)
    return host_data



@app.delete("/hosts/{host_id}")
async def remove_host(host_id: str):
    if host_id in host_monitor.hosts:
        del host_monitor.hosts[host_id]
        host_monitor.save_hosts()
        await host_monitor.notify_clients()
        return {"status": "success"}
    return {"status": "not found"}

@app.put("/hosts/{host_id}/toggle-monitoring")
async def toggle_host_monitoring(host_id: str):
    if host_id in host_monitor.hosts:
        host = host_monitor.hosts[host_id]
        host.is_monitoring = not host.is_monitoring
        await host_monitor.notify_clients()
        return host
    return None

@app.put("/hosts/{host_id}/notification-mode")
async def set_notification_mode(host_id: str, mode: str = fastapi.Query(...)):
    host = host_monitor.hosts.get(host_id)
    if not host:
        raise fastapi.HTTPException(status_code=404, detail="Host not found")
    valid_modes = {'notify_up', 'notify_down', 'disabled'}
    if mode not in valid_modes:
        raise fastapi.HTTPException(status_code=400, detail="Invalid mode")

    host.notification_mode = mode

    if mode == 'notify_up' and host.is_up:
        host.last_status_change = datetime.now()
    elif mode == 'notify_down' and not host.is_up:
        host.last_status_change = datetime.now()
    
    await host_monitor.notify_clients()
    return {"status": "success"}

@app.get("/")
async def serve_frontend():
    return FileResponse('index.html')

@app.get("/hosts/{host_id}/details")
def get_host_logs(host_id: str):
    host = host_monitor.hosts.get(host_id)
    if not host:
        return {"error": "Host not found"}, 404

    valid_latencies = [lat for lat in host.latency_history if lat is not None]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else None

    return {
        "id": host.id,
        "address": host.address,
        "is_up": host.is_up,
        "is_monitoring": host.is_monitoring,
        "statistics": {
            "success_rate": sum(host.ping_history) / len(host.ping_history) * 100 if host.ping_history else 0,
            "average_latency": avg_latency,
        },
        "response_log": host.response_log,
    }

async def main():
    logger.debug("Initializing monitoring system")
    loop = asyncio.get_running_loop()
    
    server_config = uvicorn.Config(
        app, 
        host='0.0.0.0', 
        port=9123, 
        log_level='info'
    )
    server = uvicorn.Server(server_config)
    
    shutdown_event = asyncio.Event()

    def signal_handler(*_):
        logger.info("\nShutting down...")
        shutdown_event.set()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, signal_handler)
    else:
        import win32api
        def windows_handler(type):
            signal_handler()
            return True
        win32api.SetConsoleCtrlHandler(windows_handler, True)

    # Start server as a task
    logger.debug("Starting server task")
    server_task = asyncio.create_task(server.serve())

    # Start monitoring tasks
    logger.debug("Starting monitoring tasks")
    monitor_task = asyncio.create_task(host_monitor.monitor_hosts(shutdown_event))
    host_monitor.shutdown_event = shutdown_event
    
    logger.debug("Starting dns tasks")
    dns_task = asyncio.create_task(host_monitor.resolve_dns_for_all_hosts(shutdown_event))
    
    # Wait for shutdown signal
    await shutdown_event.wait()

    # Trigger server shutdown
    server.should_exit = True
    await server_task

    # Cancel monitoring tasks
    tasks = [monitor_task, dns_task]
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.info("Clean shutdown complete")

# Configurar el event loop adecuado en Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    asyncio.run(main())