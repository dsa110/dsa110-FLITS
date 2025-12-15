
import nbformat

nb_path = 'scattering/notebooks/run_scat_analysis_walkthrough.ipynb'
print(f"Loading notebook from {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

target_source_part = "config_path = os.path.join(cwd, 'scattering/configs/bursts/dsa/freya_dsa.yaml')"
replacement_source = """cwd = os.getcwd()
# Check if we are in the notebooks directory and adjust path accordingly
if os.path.basename(cwd) == 'notebooks':
    config_path = os.path.join(cwd, '../configs/bursts/dsa/freya_dsa.yaml')
else:
    config_path = os.path.join(cwd, 'scattering/configs/bursts/dsa/freya_dsa.yaml')"""

found = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if target_source_part in cell.source:
            print("Found target cell. Updating content...")
            # We replace the specific lines but keep the rest of the cell if possible, 
            # or just replace the specific bad line + context.
            # The cell content is:
            # cwd = os.getcwd()
            # config_path = os.path.join(cwd, 'scattering/configs/bursts/dsa/freya_dsa.yaml')
            # ...
            
            # Simple string replacement for the block
            old_block = "cwd = os.getcwd()\nconfig_path = os.path.join(cwd, 'scattering/configs/bursts/dsa/freya_dsa.yaml')"
            if old_block in cell.source:
                 cell.source = cell.source.replace(old_block, replacement_source)
                 found = True
                 break

if found:
    print("Writing updated notebook...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook updated successfully.")
else:
    print("Target cell not found!")
