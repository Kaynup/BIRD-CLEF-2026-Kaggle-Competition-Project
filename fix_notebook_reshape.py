import json
import os

files_to_fix = [
    'kaggle-pipelines/pipelines/proto1/alter-strat-2/vit-geo-acoustic-inference.ipynb'
]

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "log_mels.view(len(windows)" in line:
                    line = line.replace(".view(len(windows)", ".reshape(len(windows)")
                if ".view(-1, 1, 1)" in line:
                    line = line.replace(".view(-1, 1, 1)", ".reshape(-1, 1, 1)")
                new_source.append(line)
            cell['source'] = new_source
            
    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

print("Fixed view() to reshape() in inference notebook.")
