import json
import os
import numpy as np
import re

def check_overlap(str1, str2, n):
    len1 = len(str1)
    len2 = len(str2)
    
    if str1 in str2 or str2 in str1:
        return True
    # Check overlap by comparing suffix of str1 with prefix of str2
    for i in range(1, min(len1, len2) + 1):
        if i > n and str1[-i:] == str2[:i]:
            return True
    
    # Check overlap by comparing prefix of str1 with suffix of str2
    for i in range(1, min(len1, len2) + 1):
        if i > n and str1[:i] == str2[-i:]:
            return True
    
    return False

def contexts_to_segments(contexts):
    segment_size = 100
    context = contexts[0]
    words = context.split(' ')

    # Create a list to hold segments
    segments = []
    
    # Iterate over the words and group them into segments
    for i in range(0, len(words), segment_size):
        # Join a segment of 100 words and add to segments list
        segment = ' '.join(words[i:i + segment_size])+' '
        segments.append(segment)
    
    return segments

def get_gt_ids(all_texts, injected_adv):
    gt_ids =[]
    for j, segment in enumerate(all_texts):
        # In MuSiQue (Prompt Injection), injected_adv is a list of strings (the malicious prompt)
        # But in result file it might be stored differently.
        # Let's handle list or string.
        if isinstance(injected_adv, str):
            injected_adv = [injected_adv]
            
        for malicious_text in injected_adv:
            if check_overlap(segment, malicious_text, 10):
                gt_ids.append(j)
    return list(set(gt_ids))

def calculate_precision_recall_f1(predicted, actual):
    predicted_set = set(predicted)
    actual_set = set(actual)
    
    TP = len(predicted_set & actual_set)
    FP = len(predicted_set - actual_set)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def analyze_results(file_path, task_name):
    print(f"Analyzing {task_name} results from: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_precision = 0
    total_time = 0
    total_asv = 0
    count = 0
    
    for item in data:
        if 'scores' not in item or 'important_ids' not in item:
            continue
            
        scores = item['scores']
        ids = item['important_ids']
        time_taken = item.get('time', 0)
        
        K = 10 # Updated to match run_deepseek_reproduce.py
        sorted_indices = np.argsort(scores)[::-1][:K]
        top_k_ids = [ids[i] for i in sorted_indices]
        
        # Calculate Precision if missing
        if 'precision' not in item:
            # Reconstruct GT
            contexts = item['contexts']
            # For MuSiQue, explanation level is 'segment'
            all_texts = contexts_to_segments(contexts)
            gt_texts = item['gt_important_texts'] # This is the malicious prompt
            
            gt_ids = get_gt_ids(all_texts, gt_texts)
            precision = calculate_precision_recall_f1(top_k_ids, gt_ids)
            total_precision += precision
        else:
            # If precision is already in item (calculated by evaluate.py using its own K), 
            # we should re-calculate it using our new K=10 if possible.
            # However, 'item' only has the final scalar 'precision' from the run.
            # But wait, 'scores' and 'important_ids' ARE in the item.
            # And 'gt_important_texts' is in the item.
            # So we CAN recalculate precision for K=10 even if the file has old K=5 precision.
            
            # For NQ (PoisonedRAG), gt is implicitly known: first 5 are injected? 
            # Actually for PoisonedRAG, the 'injected_adv' (gt_important_texts) is list of strings.
            # We need to map them back to IDs.
            # The 'contexts' in item is list of strings.
            # So we can do the same logic as above.
            
            contexts = item['contexts']
            # Check if contexts are segments or full docs.
            # For NQ, they are full docs (list of strings).
            # For MuSiQue, they are segments.
            
            # Let's try to detect based on task name or structure
            gt_texts = item['gt_important_texts']
            
            # Map gt_texts to IDs in contexts
            gt_ids = []
            if isinstance(contexts, list):
                for i, ctx in enumerate(contexts):
                    # Check if any gt_text is in ctx
                    is_poison = False
                    if isinstance(gt_texts, list):
                        for gt in gt_texts:
                            if check_overlap(ctx, gt, 20): # simple overlap check
                                is_poison = True
                                break
                    elif isinstance(gt_texts, str):
                         if check_overlap(ctx, gt_texts, 20):
                                is_poison = True
                    
                    if is_poison:
                        gt_ids.append(i)
            
            if len(gt_ids) > 0:
                # Recalculate precision with new K
                 precision = calculate_precision_recall_f1(top_k_ids, gt_ids)
                 total_precision += precision
            else:
                # Fallback to stored precision if we can't reconstruct
                total_precision += item['precision']
            
        total_time += time_taken
        if 'asv' in item:
            total_asv += item['asv']
        count += 1
            
    if count > 0:
        print(f"Count: {count}")
        print(f"Avg Precision: {total_precision / count:.4f}")
        print(f"Avg Time: {total_time / count:.2f}s")
        if 'asv' in data[0]:
             print(f"ASR_after: {total_asv / count:.4f}")
        else:
             print("ASR_after: N/A (Requires live model query)")
    else:
        print("No valid data found.")

# Paths
musique_path = r"e:\trac\TracLLM\results\results_musique_default\default_inject_times_5_musique_deepseek-chat_tracllm_stc_loo_denoised_shapley_5.json"
nq_path = r"e:\trac\TracLLM\results\results_nq-poison_default\PoisonedRag_nq-poison_deepseek-chat_tracllm_stc_loo_denoised_shapley_5.json"

analyze_results(musique_path, "Prompt Injection (MuSiQue)")
print("-" * 30)
analyze_results(nq_path, "Knowledge Corruption (NQ)")
