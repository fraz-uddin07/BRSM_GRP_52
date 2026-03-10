import json
import glob
import math

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects_data = list(raw_data['fluency-spam'].values())

for s in subjects_data:
    if 'data' not in s: continue
    
    vft_records = {}
    spam_records = {}
    
    for t in s['data']:
        if t.get('task') == 'VFT' and 'practice' not in str(t.get('domain', '')).lower():
            dom = t.get('domain')
            tr = t.get('tagged_responses')
            rt = t.get('response_times')
            if dom and tr and rt:
                try:
                    words = [w['response'] for w in json.loads(tr)]
                    times = json.loads(rt)
                    vft_records[dom] = {'words': words, 'rt': times}
                except: pass
                
        if t.get('task') == 'SpAM' and 'practice' not in str(t.get('domain', '')).lower():
            dom = t.get('domain')
            dw = t.get('droppedwords')
            if dom and dw:
                # dw is already a list of dicts in python if parsed, let's check
                if isinstance(dw, str):
                    try: dw = json.loads(dw)
                    except: pass
                
                coords = {}
                for item in dw:
                    if isinstance(item, dict) and 'word' in item and 'x_norm' in item:
                        w = item['word']
                        # store average if dropped multiple times? Or just last?
                        coords[w] = (item['x_norm'], item['y_norm'])
                spam_records[dom] = coords
                
    if vft_records and spam_records:
        print("Found a subject with both VFT and SpAM!")
        for dom in vft_records.keys():
            if dom in spam_records:
                print(f"Domain: {dom}")
                words = vft_records[dom]['words']
                rts = vft_records[dom]['rt']
                coords = spam_records[dom]
                print(f"  VFT words: {len(words)}, SpAM words: {len(coords)}")
                print(f"  First 3 VFT words: {words[:3]}")
                print(f"  First 3 RTs: {rts[:3]}")
                
                # compute consecutive distances
                print("  Consecutive IRTs and Distances:")
                for i in range(1, min(len(words), 5)):
                    w1, w2 = words[i-1], words[i]
                    irt = rts[i]
                    if w1 in coords and w2 in coords:
                        c1, c2 = coords[w1], coords[w2]
                        dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                        print(f"    {w1} -> {w2} | IRT: {irt:.1f}ms | Dist: {dist:.3f}")
        break

import os
metrics_file = 'all_metrics.json'
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    except:
        all_metrics = {}
else:
    all_metrics = {}

out_metrics = {}
if vft_records and spam_records:
    out_metrics["subject_found"] = True
    for dom in vft_records.keys():
        if dom in spam_records:
            words = vft_records[dom]['words']
            rts = vft_records[dom]['rt']
            coords = spam_records[dom]
            
            consecutive = []
            for i in range(1, min(len(words), 5)):
                w1, w2 = words[i-1], words[i]
                irt = rts[i]
                if w1 in coords and w2 in coords:
                    c1, c2 = coords[w1], coords[w2]
                    dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    consecutive.append({"w1": w1, "w2": w2, "irt_ms": irt, "dist": dist})
                    
            out_metrics[dom] = {
                "vft_words": len(words),
                "spam_words": len(coords),
                "sample_consecutive_distances": consecutive
            }

all_metrics["explore_spam.py"] = out_metrics

with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=4)
