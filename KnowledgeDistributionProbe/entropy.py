import gc
import os
import pickle
from typing import List

import numpy as np
import torch.nn.functional as F

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config
from dataset import EmbeddingDataset
from utils.log_util import logger
from utils.save_utils import save_ckpt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


class EntailmentDebertaBatch:
    def __init__(self, batch_size=256):
        self.tokenizer = AutoTokenizer.from_pretrained(config.ENTAILMENT_DEBERTA_PATH, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.ENTAILMENT_DEBERTA_PATH, trust_remote_code=True).to(DEVICE)
        self.model.eval()
        self.batch_size = min(batch_size, 256)

    def check_implication_batch(self, text_pairs: List[tuple[str, str]]):
        """
        批量检查蕴含关系。
        """
        results = []
        batch_inputs = text_pairs

        # 对未缓存的批次进行处理
        if batch_inputs:
            # 分批处理
            for i in tqdm(range(0, len(batch_inputs), self.batch_size), desc="Checking Entailment Inner Progress"):
                batch = batch_inputs[i:i + self.batch_size]
                texts1, texts2 = zip(*batch)  # 解压缩成两个列表
                inputs = self.tokenizer(list(texts1), list(texts2), return_tensors="pt", padding=True, truncation=True,
                                        max_length=256).to(DEVICE)

                with torch.no_grad():  # 推理时不需要计算梯度
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predictions = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().tolist()

                results.extend(predictions)

        return results

    def check_implication(self, text1, text2):
        # 包装成单例的批处理
        return self.check_implication_batch([(text1, text2)])[0]


def get_semantic_ids_batch(strings_list_batch: List[List[str]], model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    semantic_set_ids_batch = []
    # Initialise all ids with -1.
    for i in range(len(strings_list_batch)):
        semantic_set_ids_batch.append([-1] * len(strings_list_batch[i]))
    # Keep track of current id.
    next_id_batch = [0] * len(strings_list_batch)
    # 确保semantic_set_ids_batch中每个list的长度都是相同的
    assert len(set([len(ids) for ids in semantic_set_ids_batch])) == 1
    if len(strings_list_batch) == 0:
        return []

    # 输入给deberta模型
    tmp_equivalent_check_list = []
    # 用来标记上面的list中的每个元素对应的idx和j
    idx_i_j_map = []

    for i in range(len(strings_list_batch[0])):
        for idx, strings_list in enumerate(strings_list_batch):
            # Check if string1 already has an id assigned.
            if semantic_set_ids_batch[idx][i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                for j in range(i + 1, len(strings_list)):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    tmp_equivalent_check_list.append((strings_list[i], strings_list[j]))
                    idx_i_j_map.append((idx, i, j))
                    tmp_equivalent_check_list.append((strings_list[j], strings_list[i]))
                    idx_i_j_map.append((idx, j, i))

        entilement_results_list = model.check_implication_batch(tmp_equivalent_check_list)

        entilement_results_dict = {
            f"{idx}_{i}_{j}": result for (idx, i, j), result in zip(idx_i_j_map, entilement_results_list)
        }

        for idx, strings_list in enumerate(strings_list_batch):
            if semantic_set_ids_batch[idx][i] == -1:
                semantic_set_ids_batch[idx][i] = next_id_batch[idx]
                for j in range(i + 1, len(strings_list)):
                    if strict_entailment:
                        semantically_equivalent = entilement_results_dict[f"{idx}_{i}_{j}"] == 2 and \
                                                  entilement_results_dict[f"{idx}_{j}_{i}"] == 2

                    else:
                        implications = [entilement_results_dict[f"{idx}_{i}_{j}"],
                                        entilement_results_dict[f"{idx}_{j}_{i}"]]
                        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

                    if semantically_equivalent:
                        semantic_set_ids_batch[idx][j] = next_id_batch[idx]
                next_id_batch[idx] += 1

        tmp_equivalent_check_list.clear()
        idx_i_j_map.clear()
        entilement_results_dict = {}
    return semantic_set_ids_batch


def compute_semantic_entropy(embedding_dataset: EmbeddingDataset, *,
                             batch_size=256,
                             start_idx: int = 0
                             ):
    """
    Compute the semantic entropy of the generated answers.
    """

    entailment_model = EntailmentDebertaBatch()
    for idx in tqdm(range(start_idx, len(embedding_dataset), batch_size), desc="Computing SE Total Progress"):
        gc.collect()
        if idx % 1024 == 0 and idx > start_idx:
            save_ckpt(embedding_dataset, f"entropy_result", idx)
        data_batch = embedding_dataset[idx:idx + batch_size]
        responses_batch = [[data["question"] + " " + answer["text"] for answer in data["full_answers"]] for data in
                           data_batch]
        # Compute semantic ids.
        semantic_ids_batch = get_semantic_ids_batch(
            responses_batch, model=entailment_model,
            strict_entailment=True, example=None)
        # Compute semantic entropy.
        for i, data in enumerate(data_batch):
            data["semantic_ids"] = semantic_ids_batch[i]
            data["cluster_assignment_entropy"] = cluster_assignment_entropy(semantic_ids_batch[i])
        logger.info(
            f"Question ID: {data_batch[0]['id']}, Cluster Assignment Entropy: {data_batch[0]['cluster_assignment_entropy']}")
        torch.cuda.empty_cache()

    save_ckpt(embedding_dataset, f"entropy_result", len(embedding_dataset))
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    with open(os.path.join(config.SAVE_PATH, "test.pkl"), "rb") as f:
        result = pickle.load(f)
    compute_semantic_entropy(result)
    with open(os.path.join(config.SAVE_PATH, "test-entropy-result.pkl"), "wb") as f:
        pickle.dump(result, f)
