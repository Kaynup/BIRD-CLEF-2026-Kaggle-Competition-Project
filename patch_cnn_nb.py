import json

nb_path = '/home/legionlinux/miniconda3/envs/torchenv/__INIT__/Kaggle/birdclef-2026/kaggle-notebooks/birdclef-2026-cnn.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'num_workers=2' in line:
                source[i] = line.replace('num_workers=2', 'num_workers=0')
            
            # The user requested: "After training the modell save the model weigths also in the output in /kaggle/working/"
            if 'return val_preds, val_df_fold' in line:
                # Add an explicit save of the model before returning from train_fold
                indent = line[:len(line) - len(line.lstrip())]
                new_lines = [
                    f"{indent}# Save model explicitly to /kaggle/working as requested\n",
                    f"{indent}torch.save(model.state_dict(), f'/kaggle/working/birdclef_model_fold_{{fold}}.pth')\n",
                    line
                ]
                source[i] = ''.join(new_lines)
            
            # Just in case `fold_{fold}_best.pth` was not fully what they meant, let's keep the best logic.

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
