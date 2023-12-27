import os
import shutil
from tqdm import trange, tqdm

for name in ["toydesk1", "toydesk2", "scannet0024", "scannet0038", "scannet0113", "scannet0192", "scannet0113_multi"]:

    root1 = f'{os.getcwd()}/preprocess/{name}_lamaout'
    root2 = f'{os.getcwd()}/preprocess/{name}_lamaout_bg/full'
    os.makedirs(root2, exist_ok=True)

    frames = [i.split('_')[0] for i in sorted(os.listdir(root1))]
    print('%d frames'%len(frames))

    for frame in tqdm(frames):
        if name.startswith('scannet') and 'multi' not in name:
            path1 = os.path.join(root1, frame+'_0001_mask.png')
        else:
            path1 = os.path.join(root1, frame+'_0000_mask.png')
        path2 = os.path.join(root2, frame+'.jpg')

        assert os.path.exists(path1), path1
        # if not os.path.exists(path2):
        shutil.copy(path1, path2)
