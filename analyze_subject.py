import json
import glob

files = glob.glob('responses*.json')
raw_data = json.load(open(files[0], encoding='utf-8'))
subjects = list(raw_data['fluency-spam'].values())

for s in subjects:
    if 'data' in s and len(s['data']) > 0:
        print(f"Subject data length: {len(s['data'])}")
        for idx, trial in enumerate(s['data']):
            print(f"--- Trial {idx} ---")
            print(f"trial_type: {trial.get('trial_type')}")
            print(f"typeoftrial: {trial.get('typeoftrial')}")
            print(f"task: {trial.get('task')}")
            
            resp = trial.get('response')
            if isinstance(resp, dict):
                # Only take the first few keys if it's huge
                print(f"response keys: {list(resp.keys())}")
                if 'response' in resp:
                   print(f"nested response: {resp['response']}")
            else:
                print(f"response: {resp}")
                
            if 'droppedwords' in trial:
                words = [w.get('word') for w in trial['droppedwords']]
                print(f"droppedwords: {words}")
            if 'words' in trial:
                print(f"words param: {trial['words']}")
        
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

        all_metrics["analyze_subject.py"] = {
            "subject_data_length": len(s['data'])
        }

        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
            
        break
