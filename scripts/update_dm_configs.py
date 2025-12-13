#!/usr/bin/env python
"""
Update batch configuration files to use catalog DM values from bursts.yaml

This script reads DM values from bursts.yaml and updates all batch config files
in batch_configs/{chime,dsa}/ to use the correct dm_init values instead of 0.0.
"""

from pathlib import Path
import yaml

def update_batch_configs():
    """Update all batch configs with catalog DM values."""
    
    # Load bursts.yaml
    bursts_yaml = Path('bursts.yaml')
    if not bursts_yaml.exists():
        print(f"Error: {bursts_yaml} not found")
        return
    
    with open(bursts_yaml) as f:
        burst_data = yaml.safe_load(f)
    
    bursts = burst_data.get('bursts', {})
    
    # Update configs for each burst and telescope
    updated_count = 0
    for burst_name, metadata in bursts.items():
        dm_catalog = metadata.get('dm')
        if dm_catalog is None:
            print(f"Warning: No DM found for {burst_name}, skipping")
            continue
        
        for telescope in ['chime', 'dsa']:
            config_path = Path(f'batch_configs/{telescope}/{burst_name}_{telescope}.yaml')
            
            if not config_path.exists():
                print(f"Warning: {config_path} not found, skipping")
                continue
            
            # Read config
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Update dm_init
            old_dm = config.get('dm_init', 0.0)
            config['dm_init'] = float(dm_catalog)
            
            # Write back
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✓ Updated {config_path.name}: dm_init {old_dm:.1f} → {dm_catalog:.3f}")
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count} config files")


if __name__ == '__main__':
    update_batch_configs()
