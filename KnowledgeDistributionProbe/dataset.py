import json
import random

import torch
from torch.utils.data import Dataset
import datasets

import config
from utils.save_utils import md5hash


class ChoiceDataset(Dataset):
    def __init__(self, data_list: list, *, max_len=None, sample_num=5, random_seed=42):
        random.seed(random_seed)
        self.sample_list = []
        self.sample_num = sample_num
        self.random_seed = random_seed
        if (max_len is not None) and (max_len < len(data_list)):
            self.data_list = data_list[:max_len]
        else:
            self.data_list = data_list

        if len(data_list) - len(self.data_list) >= sample_num:
            self.sample_list = random.sample(data_list, sample_num)
        else:
            self.sample_list = random.sample(self.data_list, sample_num)
            self.data_list = [d for d in self.data_list if d not in self.sample_list]

        for data in self.data_list + self.sample_list:
            options = data['options']
            options.append(f"{chr(ord('A') + len(options))}：以上都不对")

        self.base_prompt = f"请回答下面的不定项选择题，选择所有最合适的选项，仅连续输出选项开头的大写英文字母，不要有标点符号或其他字符\n"
        for sample in self.sample_list:
            self.base_prompt += ChoiceDataset._make_prompt(sample['query'], sample['options'], sample['answer'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if isinstance(idx, int):
            data: dict = self.data_list[idx]
            question = ChoiceDataset._make_prompt(data['query'], data['options'])
            prompt = self.base_prompt + question

            return {
                "id": data['id'],
                "prompt": prompt,
                "question": question,
                "answer": data['answer']
            }
        elif isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data_list)))]
        else:
            raise TypeError("Invalid index type: {}".format(type(idx)))

    @staticmethod
    def _make_prompt(question: str, choices: list, answer: str = None):
        tmp = '\n'
        prompt = (
            f"问题：{question}\n"
            f"选项：\n"
            f"{tmp.join(choices)}\n"
            f"答案：{answer + tmp * 2 if answer is not None else ''}")

        return prompt


class QADataset(Dataset):
    def __init__(self, data_list: list, *, max_len=None, random_seed=42):
        random.seed(random_seed)
        self.random_seed = random_seed
        if (max_len is not None) and (max_len < len(data_list)):
            self.data_list = data_list[:max_len]
        else:
            self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.data_list[idx]
            return {
                "id": data['id'],
                "prompt": self._make_prompt(data),
                "question": data['query'],
                "answer": data['answer']
            }
        elif isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data_list)))]
        else:
            raise TypeError("Invalid index type: {}".format(type(idx)))

    def _make_prompt(self, data: dict):
        return f"请用短语或一段话简洁地回答下面的问题，用中文，不要换行，不要带标题\n问题：{data['query']}\n回答："


class QADatasetWithContext(QADataset):
    def _make_prompt(self, data: dict):
        return f"请用短语或一段话简洁地回答下面的问题，用中文，不要换行，不要带标题\n上下文：{data['context']}\n问题：{data['query']}\n回答："


# Create Dataset class for easier attribute keeping
class ClassifierDataset(Dataset):
    def __init__(self, entropy_result_list, embedding_dataset=None):
        self.name = config.RUN_NAME + "_classifier_train_dataset"
        self.question_ids = list([result['id'] for result in entropy_result_list])
        # self.question = list([result['question'] for result in entropy_result_list])
        # self.generated_text = list([result['generated_text'] for result in entropy_result_list])
        # self.full_answers = list([result['full_answers'] for result in entropy_result_list])
        # self.ground_truth = list([result['ground_truth'] for result in entropy_result_list])
        if embedding_dataset is None:
            self.tbg_dataset = torch.stack(
                [result['hidden_last_tok_before_gen'] for result in entropy_result_list]).squeeze(1).transpose(0, 1).to(
                torch.float32).cpu()
        else:
            self.tbg_dataset = torch.stack(
                [result['hidden_last_tok_before_gen'] for result in embedding_dataset]).squeeze(1).transpose(0, 1).to(
                torch.float32).cpu()
        # self.slt_dataset = torch.stack([entropy_result_dict[qid]['hidden_tok_before_eos'] for qid in
        #                                 self.question_ids]).squeeze(1).transpose(0, 1).to(torch.float32)
        self.entropy = torch.Tensor([result['cluster_assignment_entropy'] for result in entropy_result_list]).to(
            torch.float32).cpu()
        self.accuracies = torch.Tensor([result['accuracy'] for result in entropy_result_list]).to(
            torch.float32).cpu()
        self.p_true_fixed = torch.Tensor([result['p_true_fixed'] for result in entropy_result_list]).to(
            torch.float32).cpu()
        self.p_false_fixed = torch.Tensor([result['p_false_fixed'] for result in entropy_result_list]).to(
            torch.float32).cpu()
        self.log_likelihood = [result['log_likelihood'] for result in entropy_result_list]
        self.full_answers = [result['full_answers'] for result in entropy_result_list]

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        return {
            'tbg': self.tbg_dataset[idx],
            # 'slt': self.slt_dataset[idx],
            'entropy': self.entropy[idx],
            'accuracy': self.accuracies[idx],
            'p_true_fixed': self.p_true_fixed[idx],
            'p_false_fixed': self.p_false_fixed[idx],
        }

    def to_json(self) -> str:
        # clone self
        buffer = self.__dict__.copy()

        for key in buffer.__dict__:
            if isinstance(buffer.__dict__[key], torch.Tensor):
                buffer.__dict__[key] = buffer.__dict__[key].cpu().numpy().tolist()
            if type(buffer.__dict__[key]) not in [int, float, str, list, dict]:
                buffer.__dict__[key] = str(buffer.__dict__[key])
        return json.dumps(buffer.__dict__, ensure_ascii=False, indent=2)


class EmbeddingDataset(Dataset):
    def __init__(self):
        self.data = list()
        self.ids_to_idx = dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def append(self, *, id, prompt, question, generated_text, full_answers, hidden_last_tok_before_gen, ground_truth,
               log_likelihood=None, mean_pooled_embedding=None):
        if log_likelihood is None:
            log_likelihood = []
        self.data.append({
            'id': id,
            'prompt': prompt,
            'question': question,
            'generated_text': generated_text,
            'full_answers': full_answers,
            'hidden_last_tok_before_gen': hidden_last_tok_before_gen,
            'mean_pooled_embedding': mean_pooled_embedding,
            'ground_truth': ground_truth,
            'log_likelihood': log_likelihood
        })
        self.ids_to_idx[id] = len(self.data) - 1

    def del_hidden_states(self):
        for idx in range(len(self.data)):
            if 'hidden_last_tok_before_gen' in self.data[idx]:
                del self.data[idx]['hidden_last_tok_before_gen']
            if 'mean_pooled_embedding' in self.data[idx]:
                del self.data[idx]['mean_pooled_embedding']


class EnglishQADatasetBase(Dataset):
    def __init__(self, dataset_path, *, max_len=None, sample_num=5, random_seed=42, split="train"):
        random.seed(random_seed)
        self.sample_list = []
        self.sample_num = sample_num
        self.random_seed = random_seed
        data_list = datasets.load_dataset(dataset_path)[split]

        if (max_len is not None) and (max_len < len(data_list)):
            data_list = data_list[:max_len + sample_num]

        self.data_list = self._convert_data_list(data_list)

        if len(data_list) - len(self.data_list) >= sample_num:
            self.sample_list = random.sample(data_list, sample_num)
        else:
            self.sample_list = random.sample(self.data_list, sample_num)
            self.data_list = [d for d in self.data_list if d not in self.sample_list]

        self.base_prompt = f"Answer the following question as briefly as possible.\n"
        for sample in self.sample_list:
            self.base_prompt += self._make_prompt(sample['query'], sample['answer'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if isinstance(idx, int):
            data: dict = self.data_list[idx]
            prompt = self.base_prompt + self._make_prompt(data['query'], None)

            return {
                "id": data['id'],
                "prompt": prompt,
                "question": data['query'],
                "answer": data['answer']
            }
        elif isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data_list)))]
        else:
            raise TypeError("Invalid index type: {}".format(type(idx)))

    def _make_prompt(self, question: str, answer: str = None):
        tmp = '\n'
        prompt = (
            f"Question：{question}\n"
            f"Answer：{answer + tmp * 2 if answer is not None else ''}")

        return prompt

    def _convert_data_list(self, data_list):
        raise NotImplementedError()


class TriviaQADataset(EnglishQADatasetBase):
    def _convert_data_list(self, data_list):
        converted_data_list = []
        for i in range(len(data_list['question_id'])):
            converted_data_list.append({
                "id": data_list['question_id'][i],
                "query": data_list['question'][i],
                "answer": data_list['answer'][i]['value']
            })
        return converted_data_list


class NQOpenDataset(EnglishQADatasetBase):
    def _convert_data_list(self, data_list):
        converted_data_list = []
        for i in range(len(data_list['question'])):
            converted_data_list.append({
                "id": f"nq_open_{md5hash(data_list['question'][i])}",
                "query": data_list['question'][i][-512:],
                "answer": data_list['answer'][i][0][:512]
            })
        return converted_data_list


class MedMCQADataset(EnglishQADatasetBase):
    def _convert_data_list(self, data_list):
        converted_data_list = []
        for i in range(len(data_list['question'])):
            choice = chr(data_list['cop'][i] + ord('a'))
            converted_data_list.append({
                "id": f"medmcqa_{data_list['id'][i]}",
                "query": data_list['question'][i][-512:],
                "answer": data_list[f"op{choice}"][i][:512]  # 单选选项作为标答
            })
        return converted_data_list
