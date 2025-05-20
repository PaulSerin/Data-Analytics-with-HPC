import torch

if torch.cuda.is_available():
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    total_mem = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
    print(f"✅ GPU disponible : {gpu_name} (ID: {gpu_index})")
    print(f"🧠 Mémoire totale : {total_mem:.2f} Go")
else:
    print("❌ Aucun GPU CUDA disponible.")
