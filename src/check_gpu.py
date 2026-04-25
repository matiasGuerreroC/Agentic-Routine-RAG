import torch
print(f"¿Pytorch puede usar CUDA?: {torch.cuda.is_available()}")
print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")