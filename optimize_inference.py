import json
import os

file_path = 'kaggle-pipelines/pipelines/proto1/alter-strat-2/vit-geo-acoustic-inference.ipynb'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # The optimization: We need to replace the entire '2. VRAM-Safe Sub-Batched Inference' block
            if "with torch.no_grad():" in source and "mel_specs = mel_transform(waveforms)" in source:
                
                new_inference_block = """    # 2. Fully Streamed VRAM & RAM Safe Inference
    ac_out_list, geo_out_list = [], []
    sub_batch_size = 64
    
    # Pre-allocate mean and std on GPU
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    # Enable all CPU cores for KNN if it exists
    if knn_model is not None:
        knn_model.n_jobs = -1
        
    for i in range(0, len(windows), sub_batch_size):
        sub_wins = windows[i:i+sub_batch_size]
        sub_waveforms = torch.tensor(np.array(sub_wins), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            sub_mels = mel_transform(sub_waveforms)
            log_mels = torch.log(sub_mels + 1e-6)
            
            m_min = log_mels.reshape(len(sub_wins), -1).min(dim=1)[0].reshape(-1, 1, 1)
            m_max = log_mels.reshape(len(sub_wins), -1).max(dim=1)[0].reshape(-1, 1, 1)
            log_mels = (log_mels - m_min) / (m_max - m_min + 1e-6)
            
            sub_images = log_mels.unsqueeze(1)
            sub_images = F.interpolate(sub_images, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
            sub_images = (sub_images - mean) / std
            
            ac_out_sub, geo_out_sub = model(sub_images)
            
            ac_out_list.append(torch.sigmoid(ac_out_sub).cpu().numpy())
            geo_out_list.append(geo_out_sub.cpu().numpy())
            
    probs_visual = np.concatenate(ac_out_list, axis=0)
    predicted_coords = np.concatenate(geo_out_list, axis=0)
    
    # 3. Dynamic Self-Distillation Spatial Prior (Vectorized for Speed)
    if knn_model is not None:
        probs_knn = knn_model.predict(predicted_coords)
        max_probs = np.max(probs_visual, axis=1, keepdims=True)
        mask = (max_probs > CFG.CONFIDENCE_THRESHOLD)
        final_probs = np.where(mask, probs_visual, probs_visual * (1 - CFG.KNN_ALPHA) + probs_knn * CFG.KNN_ALPHA)
    else:
        final_probs = probs_visual
"""
                
                # Split the cell source by the start of the inference block and the end of the gating logic
                part1 = source.split("waveforms = torch.tensor(np.array(windows), dtype=torch.float32).to(device)")[0]
                part2 = source.split("final_probs = probs_visual")[1]
                
                new_source = part1 + new_inference_block + part2
                cell['source'] = [line + "\\n" for line in new_source.split("\\n") if line]

    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

print("Optimized inference notebook for RAM swap thrashing and CPU parallelization.")
