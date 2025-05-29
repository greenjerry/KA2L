import gc
import os
import sys

import numpy as np
from torch import softmax
from tqdm import tqdm
from transformers import StoppingCriteriaList

import config
from utils import args_util, log_util
from utils.log_util import logger
from utils.model_util import mean_pool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle

import torch

from dataset import ChoiceDataset, QADataset, QADatasetWithContext, TriviaQADataset, EmbeddingDataset, NQOpenDataset, \
    MedMCQADataset
from model import ModelAndTokenizer, StoppingCriteriaSub, STOP_SEQUENCES, \
    LlamaMLPClassifier

torch.set_grad_enabled(False)


def get_hidden_states(mt: ModelAndTokenizer,
                      dataset,
                      max_input_length: int = 512,
                      max_length: int = 256,
                      batch_size: int = 16,
                      temperature: float = 0.7,
                      start_idx: int = 0,
                      ):
    """
    通过generate的hidden states来生成中间状态，不包括其他模块的中间状态
    可以设定batch_size以加速
    :param dataset:
    :param max_input_length:
    :param mt:
    :param temperature:
    :param top_k:
    :param max_length:
    :param batch_size:
    :return:
    """

    result: EmbeddingDataset = EmbeddingDataset()

    for idx in tqdm(range(start_idx, len(dataset), batch_size), desc="Sampling First with Hidden States"):
        batch_dataset = dataset[idx:idx + batch_size]
        batch_prompts = [d["prompt"] for d in batch_dataset]
        current_batch_length = len(batch_dataset)
        inputs = mt.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(mt.device)
        inputs["input_ids"] = inputs["input_ids"][:, -max_input_length:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_length:]

        input_max_lengths = inputs['input_ids'].size(1)
        # attention mask得到实际有效输入长度
        input_length_list = torch.sum(inputs["attention_mask"], dim=1).tolist()
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=STOP_SEQUENCES,
            initial_length=input_max_lengths,
            tokenizer=mt.tokenizer)])
        with torch.inference_mode(mode=True):
            model_out = mt.model.generate(
                **inputs,
                output_hidden_states=True,
                max_length=max_length + input_max_lengths,
                # max_new_tokens=max_length,
                return_dict_in_generate=True,
                temperature=temperature,
                # top_k=top_k,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=mt.tokenizer.pad_token_id,
            )

        hidden_states = model_out.hidden_states
        output_token_max_size = len(hidden_states)
        if model_out.sequences.size(1) != output_token_max_size + input_max_lengths:
            print(f"Warning: model_out.sequences.size(1)={model_out.sequences.size(1)}, "
                  f"output_token_max_size={output_token_max_size}, input_max_lengths={input_max_lengths}")
        output_texts = mt.tokenizer.batch_decode(model_out.sequences, skip_special_tokens=True)
        last_tok_bef_gen = hidden_states[0]

        ''' 右填充

        # 1. 针对每个 tensor，提取每个样本的最后一个有效 token 的 hidden states
        last_tok_bef_gen_list = []
        for layer_tensor in last_tok_bef_gen:
            # layer_tensor 的 shape: (batch_size, seq_len, hidden_dim)
            # 使用列表推导式和索引提取每个样本的最后一个有效 token 的 hidden states
            temp_list = [layer_tensor[i, input_length_list[i] - 1, :] for i in range(current_batch_length)]
            # 将 temp_list 堆叠成 tensor
            last_tok_bef_gen_list.append(torch.stack(temp_list))

        # 2. 将所有层的 hidden states 堆叠起来
        # last_tok_bef_gen_list 现在是一个 list, 每个元素是一个 tensor, shape 为 (batch_size, hidden_dim)
        # 我们需要将它变成一个 tensor, shape 为 (batch_size, num_layers, hidden_dim)
        last_tok_bef_gen_embedding = torch.stack(last_tok_bef_gen_list, dim=1)
        '''
        # ''' 左填充
        mean_pool_last_tok_bef_gen = mean_pool(last_tok_bef_gen[config.EMBEDDINGS_FROM_LAYER_N],
                                               inputs["attention_mask"])
        last_tok_bef_gen_embedding = last_tok_bef_gen[config.EMBEDDINGS_FROM_LAYER_N][:, -1, :]
        # '''
        for i in range(current_batch_length):
            output_texts_tmp = output_texts[i][len(batch_prompts[i]):].split('\n')[0]

            result.append(
                id=batch_dataset[i]["id"],
                prompt=batch_dataset[i]["prompt"],
                question=batch_dataset[i]["question"],
                generated_text=output_texts_tmp,
                full_answers=list(),
                hidden_last_tok_before_gen=last_tok_bef_gen_embedding[i].clone().cpu(),
                ground_truth=batch_dataset[i]["answer"],
                log_likelihood=None,
                mean_pooled_embedding=mean_pool_last_tok_bef_gen[i].clone().cpu(),
            )
        # del model_out, hidden_states, last_tok_bef_gen, last_tok_bef_gen_list, last_tok_bef_gen_embedding
        del model_out, hidden_states, last_tok_bef_gen, last_tok_bef_gen_embedding
        gc.collect()
        # torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return result


def classify_with_mlp(dataset: EmbeddingDataset, batch_size: int = 256):
    # 调用分类器来分类
    classfier_model_save_path = os.path.join(config.SAVE_PATH, 'results', 'models',
                                             f'llama_mlp_{config.EMBEDDINGS_FROM_LAYER_N}.pt')
    if os.path.exists(classfier_model_save_path):
        classifier_model: LlamaMLPClassifier = torch.load(classfier_model_save_path)
        classifier_model.to("cuda").to(torch.bfloat16)
        classifier_model.eval()
    else:
        raise ValueError(f"No classifier_model: {classfier_model_save_path}")

    all_predictions = []  # 存储所有预测结果

    with torch.inference_mode():  # 禁用梯度计算，加速推理
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Classifying"):
            batch_dataset = dataset[idx:idx + batch_size]
            current_batch_length = len(batch_dataset)
            inputs = torch.stack([d["hidden_last_tok_before_gen"] for d in batch_dataset]).to("cuda").to(torch.bfloat16)
            inputs = inputs.view(current_batch_length, -1)  # 展平输入特征

            # 前向传播获取 logits
            logits = classifier_model(inputs)  # logits shape: [batch_size, num_classes=2]

            # 将 logits 转换为概率 (使用 softmax)
            probs = softmax(logits, dim=1)  # probs shape: [batch_size, 2]

            # 获取预测类别 (概率最大的类别)
            predictions = torch.argmax(probs, dim=1)  # predictions shape: [batch_size]

            # 将当前批次的预测结果添加到总列表中
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return np.array(all_predictions)  # 返回 NumPy 数组


def main():
    if config.DATASET_TYPE == "choice":
        with open(config.DATASET_PATH, "r") as f:
            dataset = ChoiceDataset(json.load(f), max_len=config.DATASET_DATA_SAMPLE_SIZE)
    elif config.DATASET_TYPE == "qa":
        with open(config.DATASET_PATH, "r") as f:
            dataset = QADataset(json.load(f), max_len=config.DATASET_DATA_SAMPLE_SIZE)
    elif config.DATASET_TYPE == "qa_with_context":
        with open(config.DATASET_PATH, "r") as f:
            dataset = QADatasetWithContext(json.load(f), max_len=config.DATASET_DATA_SAMPLE_SIZE)
    elif config.DATASET_TYPE == "trivia_qa":
        dataset = TriviaQADataset(config.DATASET_PATH, max_len=config.DATASET_DATA_SAMPLE_SIZE)
    elif config.DATASET_TYPE == "nq_open":
        dataset = NQOpenDataset(config.DATASET_PATH, max_len=config.DATASET_DATA_SAMPLE_SIZE)
    elif config.DATASET_TYPE == "medmcqa":
        dataset = MedMCQADataset(config.DATASET_PATH, max_len=config.DATASET_DATA_SAMPLE_SIZE)
    else:
        raise ValueError(f"Invalid dataset type: {config.DATASET_TYPE}")

    if not os.path.exists(os.path.join(config.SAVE_PATH, "cache")):
        os.makedirs(os.path.join(config.SAVE_PATH, "cache"))
    if not os.path.exists(os.path.join(config.SAVE_PATH, "cache", "inference_embedding_ds.pkl")):
        mt = ModelAndTokenizer(
            config.MODEL_NAME,
            low_cpu_mem_usage=True,
        )
        embedding_ds = get_hidden_states(mt,
                                         dataset,
                                         batch_size=config.BATCH_SIZE,
                                         max_length=256,
                                         temperature=0.1
                                         )
        with open(os.path.join(config.SAVE_PATH, "cache", "inference_embedding_ds.pkl"), "wb") as f:
            pickle.dump(embedding_ds, f)
        del mt
        gc.collect()
        torch.cuda.empty_cache()
    else:
        with open(os.path.join(config.SAVE_PATH, "cache", "inference_embedding_ds.pkl"), "rb") as f:
            embedding_ds = pickle.load(f)

    if config.EMBEDDINGS_FROM_LAYER_N == -1:
        mt = ModelAndTokenizer(
            config.MODEL_NAME,
            low_cpu_mem_usage=True,
        )
        config.EMBEDDINGS_FROM_LAYER_N = len(mt.model.model.layers)
        del mt
        gc.collect()
        torch.cuda.empty_cache()

    classify_result = classify_with_mlp(embedding_ds)
    embedding_ds.del_hidden_states()
    sure_list = []
    unsure_list = []
    for idx in range(len(embedding_ds)):
        if classify_result[idx] == 0:
            sure_list.append(embedding_ds[idx])
        else:
            unsure_list.append(embedding_ds[idx])

    with open(os.path.join(config.SAVE_PATH, "results", "inference-sure.json"), "w") as f:
        json.dump(sure_list, f)
    with open(os.path.join(config.SAVE_PATH, "results", "inference-unsure.json"), "w") as f:
        json.dump(unsure_list, f)


if __name__ == "__main__":
    parser = args_util.get_parser()
    args, unknown = parser.parse_known_args()
    if args.config is not None:
        config.update_config(args.config, continue_training=True)

    log_util.init_logger()
    logger.info('Starting new run with args: %s', args)

    main()
