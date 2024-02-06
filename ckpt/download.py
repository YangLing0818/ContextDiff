import time
from huggingface_hub import snapshot_download
repo_id = "runwayml/stable-diffusion-v1-5" #you can change the repo-id to download other models like sd-v1.4
local_dir = repo_id.split("/")[1]
cache_dir = local_dir + "/cache"
while True:
    try:
        snapshot_download(cache_dir=cache_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.model", "*.json", 
        "*.py", "*.md", "*.txt","*.ckpt","*.bin"],
        ignore_patterns=["*.msgpack","*.h5", "*.ot"],
        )
    except Exception as e :
        print(e)
        # time.sleep(5)
    else:
        print('Finished')   
        break
    #gdown https://drive.google.com/uc?id=xxx #