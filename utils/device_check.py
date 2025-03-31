import torch
from colorama import Fore, Style

def verify_cuda():
    """Verifica disponibilitatea GPU și afiseaza informatii despre hardware"""
    print(f"\n{Fore.YELLOW}{'='*50}")
    print(f"{Fore.CYAN} Hardware Configuration:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}PyTorch version: {Fore.WHITE}{torch.__version__}")
    print(f"{Fore.GREEN}CUDA available: {Fore.WHITE}{torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\n{Fore.GREEN} GPU detected: {Fore.WHITE}{torch.cuda.get_device_name(0)}")
        print(f"{Fore.GREEN} Total memory: {Fore.WHITE}{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"{Fore.GREEN} CUDA capability: {Fore.WHITE}{torch.cuda.get_device_capability()}")
    else:
        print(f"\n{Fore.RED}  No GPU detected, using CPU")
    print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")

# Ruleaza verificarea cand este importat
verify_cuda()
