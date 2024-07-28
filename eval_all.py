import os
import json
import csv
from glob import glob
import pandas as pd

def process_json_files(base_path):
    results = []
    
    # Walk through all subdirectories
    for folder_path in glob(os.path.join(base_path, "*/")):
        model_name = os.path.basename(os.path.dirname(folder_path))
        
        # Find the JSON file starting with "results"
        json_files = glob(os.path.join(folder_path, "results*.json"))
        
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            # Process the 'results' key
            if 'results' in data:
                for dataset, values in data['results'].items():
                    if 'acc,none' in values:
                        results.append({
                            'model': model_name,
                            'dataset': dataset,
                            'accuracy': round(values['acc,none']*100, 2)
                        })
    
    return results

from collections import defaultdict
results = process_json_files('.')

from collections import defaultdict

def process_results(results):
    # Group results by model
    model_data = defaultdict(list)
    for item in results:
        if item['dataset'] != 'multimedqa':
            model_data[item['model']].append(item)

    # Calculate average and create new results
    new_results = []
    for model, items in model_data.items():
        total_acc = sum(item['accuracy'] for item in items)
        avg_acc = round(total_acc / len(items), 2)
        
        # Add individual dataset results
        new_results.extend(items)
        
        # Add average entry
        new_results.append({
            'model': model,
            'dataset': 'average',
            'accuracy': avg_acc
        })
    
    return new_results

results = process_results(results)
df = pd.DataFrame(results)
# Pivot the DataFrame to get the desired format
pivot_df = df.pivot(index='model', columns='dataset', values='accuracy')

# Reset the index to make 'model' a column
pivot_df.reset_index(inplace=True)

# Rename the 'model' column to 'file_name'
pivot_df.rename(columns={'model': 'file_name'}, inplace=True)

# Display the DataFrame
pivot_df.to_csv('llm_results.csv', index=False)
