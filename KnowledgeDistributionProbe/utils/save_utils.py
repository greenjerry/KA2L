import hashlib
import json
import os
import pickle
import traceback

import torch

from KnowledgeDistributionProbe import config


def save_ckpt(data, prefix, count):
    save_dir = os.path.join(config.SAVE_PATH, 'checkpoints')
    name = f'{prefix}_{count}.pkl'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not name.endswith('.pkl'):
        name += '.pkl'
    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(data, f)
    # clean old checkpoints
    try:
        old_file_count = []
        for root, dirname, files in os.walk(save_dir):
            for file in files:
                if file.startswith(prefix) and file.endswith('.pkl') and file != name:
                    old_count = int(file.split('_')[-1].split('.')[0])
                    if old_count < count:
                        old_file_count.append(old_count)
        old_file_count.sort()
        for i in old_file_count[:-2]:
            os.remove(os.path.join(save_dir, f'{prefix}_{i}.pkl'))
    except Exception as e:
        traceback.print_exc()
        return


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # 将 Tensor 转换为列表
        elif hasattr(obj, '__dict__'):
            try:
                json.dumps(obj.__dict__)
                return obj.__dict__
            except TypeError:
                return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)