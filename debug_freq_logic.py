import yaml
from pathlib import Path
import numpy as np

def check():
    config_path = Path("scattering/configs/telescopes.yaml")
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    
    chime = configs["chime"]
    print(f"CHIME Config: f_min={chime['f_min_GHz']}, f_max={chime['f_max_GHz']}")
    
    n_ch = 32 # simulation of downsampled
    freq = np.linspace(chime["f_min_GHz"], chime["f_max_GHz"], n_ch)
    print(f"Freq[0]: {freq[0]}, Freq[-1]: {freq[-1]}")
    
    if freq[0] < freq[-1]:
        print("Freq is ASCENDING")
    else:
        print("Freq is DESCENDING")

if __name__ == "__main__":
    check()
