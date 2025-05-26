import json
import re

def append_to_info_file(data, save_path):
    with open(save_path, 'a') as f:
        f.write(f"{data}\n")

def combine_to_json(task_file_path, video_file_path, save_path):
    with open(task_file_path, 'r') as task_file:
        task_content = task_file.read()

    with open(video_file_path, 'r') as video_file:
        video_files = [line.strip() for line in video_file.readlines()]

    task_sequences = re.findall(r'\[(.*?)\]', task_content, re.DOTALL)

    tasks = []
    for sequence in task_sequences:
        print("Original sequence:", sequence) 
        cleaned_sequence = sequence.replace('\n', '')
        print("Cleaned sequence:", cleaned_sequence) 
        tasks.append(cleaned_sequence)

    combined_data = []
    for i, task_list in enumerate(tasks):
        if i < len(video_files):  
            combined_entry = {
                'video_file': video_files[i],
                'tasks': task_list
            }
            combined_data.append(combined_entry)

    json_output = json.dumps(combined_data, indent=4)

    with open(save_path, 'w') as json_file:
        json_file.write(json_output)

    print(f"JSON data has been saved to {save_path}")