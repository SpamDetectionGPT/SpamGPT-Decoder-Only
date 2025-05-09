import torch
import time
from datetime import datetime
import os
import psutil


def get_gpu_info():
    try:
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # Convert to GB
            memory_total = props.total_memory / (1024**3)                  # Convert to GB
            
            # Get GPU utilization (if available)
            try:
                utilization = torch.cuda.utilization(i)
            except:
                utilization = 0.0
            
            gpu_info.append({
                'id': i,
                'name': props.name,
                'utilization': utilization,
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved,
                'memory_total': memory_total,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def print_gpu_status(gpu_info):
    if not gpu_info:
        return
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Print timestamp
    print(torch.version.cuda)
    print(f"\nGPU Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # Print header
    print(f"{'GPU':<5} {'Name':<20} {'Util %':<10} {'Memory (GB)':<20} {'Compute Cap':<10}")
    print("-" * 100)
    
    # Print each GPU's information
    for gpu in gpu_info:
        memory_str = f"Alloc: {gpu['memory_allocated']:.1f} | Reserved: {gpu['memory_reserved']:.1f} | Total: {gpu['memory_total']:.1f}"
        
        print(f"{gpu['id']:<5} {gpu['name']:<20} {gpu['utilization']:>5.1f}% {memory_str:<20} {gpu['compute_capability']:<10}")

def main():
    try:
        while True:
            gpu_info = get_gpu_info()
            print_gpu_status(gpu_info)
            time.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main() 