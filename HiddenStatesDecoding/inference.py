import json
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from dataset import EmbeddingDataset
from .models.model_utils import mean_pool

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['VEC2TEXT_CACHE']="/users/shenrujia/yinhx/cache/inversion"
import sys
sys.path.append("/users/shenrujia/yinhx/vec2text")

from .analyze_utils import args_from_config
from .models import  InversionModel
from .models.config import InversionConfig
from .run_args import ModelArguments, DataArguments, TrainingArguments

BATCH_SIZE=32

def main():

    # name="/users/shenrujia/yinhx/saves/inversion_deepseek_qwen2.5"
    name = sys.argv[1]
    config = InversionConfig.from_pretrained(name)
    model_args = args_from_config(ModelArguments, config)
    data_args = args_from_config(DataArguments, config)
    training_args = args_from_config(TrainingArguments, config)

    model:InversionModel = InversionModel(config=config)
    model = model.__class__.from_pretrained(name)
    model.to(training_args.device)


    # with open(sys.argv[1],'rb') as f:
    #     embedding_dataset:EmbeddingDataset = pickle.load(f)

    with open(sys.argv[2],'r') as f:
        dataset=json.load(f)
    gen_kwargs= {"max_length": 128, "min_length": 1}
    pd_table = []
    json_result=[]

    for idx in tqdm(range(0,len(dataset),BATCH_SIZE)):
        batch_size=min(BATCH_SIZE,len(dataset)-idx)
        # batch_size=min(BATCH_SIZE,len(embedding_dataset)-idx)
        # inputs = {
        #     "frozen_embeddings":torch.stack([item['hidden_last_tok_before_gen'] for item in embedding_dataset[idx:idx+batch_size]]).to(training_args.device),
        # }
        inputs = model.embedder_tokenizer([item['instruction'] for item in dataset[idx:idx+batch_size]], return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=128)
        inputs['embedder_input_ids']=inputs['input_ids']
        inputs['embedder_attention_mask']=inputs['attention_mask']

        # embeddings=mean_pool(torch.stack([torch.stack([item['hidden_last_tok_before_gen'] for item in embedding_dataset[idx:idx+batch_size]])],dim=1),inputs['embedder_attention_mask'])
        # inputs['frozen_embeddings']=torch.stack([item['mean_pooled_embedding'] for item in embedding_dataset[idx:idx+batch_size]]).to(training_args.device)
        output=model.generate(inputs=inputs,generation_kwargs=gen_kwargs)

        output_str = model.tokenizer.batch_decode(output, skip_special_tokens=True)

        for i in range(idx,idx+batch_size):
            d = dataset[i]
            json_result.append({
                'id': d['custom_id'],
                'question': d['instruction'],
                'ground_truth': d['output'],
                'decode_result': output_str[i-idx]
            })
            pd_table.append([d['instruction'], output_str[i-idx]])
    df = pd.DataFrame(pd_table, columns=["Input", "Output"])
    df.to_csv(sys.argv[2][:-5]+"_decode_result.csv", index=True)
    df.to_json(sys.argv[2][:-5]+"_decode_result.json", force_ascii=False, indent=2, orient="records")

    with open(sys.argv[2][:-5]+"_out.json",'w',encoding='utf-8') as f:
        try:
            json.dump(json_result,f,ensure_ascii=False,indent=2)
        except Exception as e:
            print(f"Error dumping JSON: {e}")
            # Handle the error as needed, e.g., log it or raise an exception
            json.dump(json_result,f,ensure_ascii=True,indent=2)

if __name__ == "__main__":
    main()