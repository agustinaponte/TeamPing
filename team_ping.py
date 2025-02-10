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
import os
import socket
import platform
import psutil
import getpass

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


def get_current_user():
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"

def is_admin():
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:  # Unix/Linux
            return os.geteuid() == 0
    except Exception:
        return False

def get_network_info():
    hostname = socket.gethostname()
    ip_addresses = []
    try:
        # Se obtienen todas las direcciones IP de las interfaces del sistema
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip_addresses.append(addr.address)
    except Exception:
        pass
    port = os.getenv("FLET_SERVER_PORT") or os.getenv("PORT") or None
    return {
        "hostname": hostname,
        "ip_addresses": ip_addresses,
        "port": port,
    }

def get_cpu_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    load_avg = None
    if hasattr(os, "getloadavg"):
        load_avg = os.getloadavg()  # Devuelve una tupla: (1 min, 5 min, 15 min)
    return {
        "cpu_percent": cpu_percent,
        "load_average": load_avg,
    }

def get_uptime_info():
    boot_time = psutil.boot_time()
    system_uptime = time.time() - boot_time
    process = psutil.Process(os.getpid())
    process_uptime = time.time() - process.create_time()
    return {
        "system_uptime_seconds": system_uptime,
        "process_uptime_seconds": process_uptime,
    }

def get_memory_info():
    virtual_mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info()
    return {
        "virtual_memory": {
            "total": virtual_mem.total,
            "available": virtual_mem.available,
            "used": virtual_mem.used,
            "percent": virtual_mem.percent,
        },
        "process_memory": {
            "rss": process_mem.rss,  # Resident Set Size
            "vms": process_mem.vms,  # Virtual Memory Size
        },
    }

def get_disk_info():
    disk_usage = psutil.disk_usage("/")  # Asumiendo la partición raíz
    disk_io = psutil.disk_io_counters()
    return {
        "usage": {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "free": disk_usage.free,
            "percent": disk_usage.percent,
        },
        "io": {
            "read_bytes": disk_io.read_bytes,
            "write_bytes": disk_io.write_bytes,
            "read_count": disk_io.read_count,
            "write_count": disk_io.write_count,
        },
    }

def get_resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as bundled executable - use sys._MEIPASS
        base_path = sys._MEIPASS
    else:
        # Running in normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

html_path = get_resource_path("index.html")

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
        self.monitor_tasks = {}
        self.load_hosts(hosts_file)
        self.websocket_clients = set()
        self.executor = ThreadPoolExecutor(max_workers=100)
        self.shutdown_event = None

    async def start_monitoring(self, host, shutdown_event):
        """Start monitoring a host and store the task"""
        if host.id in self.monitor_tasks:
            return  # Already monitoring
        task = asyncio.create_task(self.host_monitoring_loop(host, shutdown_event))
        self.monitor_tasks[host.id] = task
        # Add callback to remove task from tracking when done
        task.add_done_callback(lambda _: self.monitor_tasks.pop(host.id, None))

    async def stop_monitoring(self, host_id):
        """Stop monitoring a host by cancelling its task"""
        task = self.monitor_tasks.get(host_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.monitor_tasks.pop(host_id, None)

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

            last_entries = host.response_log[-10:]  # Get last 10 entries
            last_responses = []
            for entry in reversed(last_entries):
                # Extract timestamp and success
                timestamp_str = entry.get('timestamp')
                success = entry.get('success', False)
                unix_time = None
                if timestamp_str:
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                        unix_time = int(dt.timestamp())
                    except ValueError:
                        pass  # Keep unix_time as None if parsing fails
                last_responses.append({
                    'success': success,
                    'timestamp': unix_time
                })

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
                'last_responses': last_responses
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
        """Ping host with detailed debug logging (KEEP THIS VERSION)"""
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

    async def monitor_hosts(self, shutdown_event: asyncio.Event):
        """Start monitoring for all hosts"""
        for host in self.hosts.values():
            if host.is_monitoring:
                await self.start_monitoring(host, shutdown_event)
        await shutdown_event.wait()
        # Cancel all tasks on shutdown
        tasks = list(self.monitor_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.monitor_tasks.clear()
    
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
async def add_hosts(address: str = fastapi.Form(...)):
    # Separa las direcciones basándose en espacios
    addresses = address.split()
    new_hosts = []

    for addr in addresses:
        host_id = addr.replace('.', '_')
        if host_id not in host_monitor.hosts:
            new_host = Host(id=host_id, address=addr, is_monitoring=True)
            host_monitor.hosts[host_id] = new_host
            host_monitor.save_hosts()
            if new_host.is_monitoring:
                await host_monitor.start_monitoring(new_host, host_monitor.shutdown_event)
            new_hosts.append(new_host)
    await host_monitor.notify_clients()
    return new_hosts

@app.get("/hosts")
def get_hosts():
    host_data = []
    for host in host_monitor.hosts.values():
        valid_latencies = [lat for lat in host.latency_history if lat is not None]
        ping_history = host.ping_history

        last_responses = []
        for i in range(max(0, len(ping_history) - 10), len(ping_history)):
            response = {
                'success': ping_history[i],
                'latency': host.latency_history[i] if i < len(host.latency_history) else None
            }
            last_responses.append(response)

        data = {
            'id': host.id,
            'address': host.address,
            'dns_info': host.dns_info,
            'is_up': host.is_up,
            'is_monitoring': host.is_monitoring,
            'ping_success_rate': (sum(ping_history) / len(ping_history) * 100) if ping_history else 0,
            'avg_latency': sum(valid_latencies) / len(valid_latencies) if valid_latencies else None,
            'last_checked': host.last_checked.isoformat() if host.last_checked else None,
            'last_responses': last_responses
        }
        host_data.append(data)
    return host_data

@app.delete("/hosts/{host_id}")
async def remove_host(host_id: str):
    if host_id in host_monitor.hosts:
        # Stop monitoring before deletion
        await host_monitor.stop_monitoring(host_id)
        del host_monitor.hosts[host_id]
        host_monitor.save_hosts()
        await host_monitor.notify_clients()
        return {"status": "success"}
    return {"status": "not found"}

@app.put("/hosts/{host_id}/toggle-monitoring")
async def toggle_host_monitoring(host_id: str):
    host = host_monitor.hosts.get(host_id)
    if not host:
        return None
    
    new_state = not host.is_monitoring
    host.is_monitoring = new_state
    
    if new_state:
        await host_monitor.start_monitoring(host, host_monitor.shutdown_event)
    else:
        await host_monitor.stop_monitoring(host_id)
    
    await host_monitor.notify_clients()
    return host

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
    return FileResponse(html_path)

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

@app.get("/server-info")
async def server_info():
    info = {
        "user": get_current_user(),
        "is_admin": is_admin(),
        "network": get_network_info(),
        "cpu": get_cpu_info(),
        "uptime": get_uptime_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    }
    return info

async def main():
    logger.debug("Initializing monitoring system")
    loop = asyncio.get_running_loop()
    


    if not os.path.exists(html_path):
        logger.critical(f"Missing index.html at: {html_path}")
        sys.exit(1)

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
    input("Press Enter to exit...")

# Configurar el event loop adecuado en Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    asyncio.run(main())