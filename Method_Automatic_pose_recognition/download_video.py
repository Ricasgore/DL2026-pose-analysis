import os
import json
import yt_dlp

directory = '/home/amine_tsp/DL2026/Datasets/ActivityNet/raw_clips'

with open('/home/amine_tsp/DL2026/Datasets/ActivityNet/Evaluation/data/activity_net.v1-3.min.json') as data_file:    
    data = json.load(data_file)

videos = data['database']

# yt-dlp options
ydl_opts = {
    'format': 'best',
    'quiet': True,
    'ignoreerrors': True,
    'no_warnings': True,
    'nocheckcertificate': True,
}

counter = 0

for key, video in videos.items():
    subset = video['subset']
    annotations = video['annotations']
    
    # Process label
    label = ''
    if len(annotations) != 0:
        label = annotations[0]['label']
        label = '/' + label.replace(' ', '_')

    # Create folder
    label_dir = os.path.join(directory, subset + label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Output template configuration
    current_opts = ydl_opts.copy()
    current_opts['outtmpl'] = f'{label_dir}/{key}.%(ext)s'

    url = video['url']
    
    print(f'Downloading {counter}: {key}...')
    
    try:
        with yt_dlp.YoutubeDL(current_opts) as ydl:
            ydl.download([url])
        counter += 1
    except Exception as e:
        print(f"Failed to download {key}: {e}")