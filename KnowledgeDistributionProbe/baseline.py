import gc
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from vllm import LLM, SamplingParams

import config
from dataset import EmbeddingDataset
from model import ModelAndTokenizer

from torch.nn.functional import cross_entropy

from utils.log_util import logger
from utils.save_utils import save_ckpt


def construct_p_true_prompt(
        question: str,
        answer: str,
        full_answers: List[str],
        is_correct: bool = None,
) -> str:
    """
    Construct the prompt for P-true uncertainty metric.
    :param question:
    :param answer: 单次生成的答案，temperature=0.1
    :param full_answers: 生成10次的答案，temperature=1.0
    :param is_correct: 如果是构建few-shot prompt，则需要传入，否则传入None
    :return:
    """
    prompt = ["Question: " + question, "\nBrainstormed Answers: "]
    for full_answer in full_answers:
        prompt.append(full_answer.strip() + "\n")
    prompt.append("Possible answer: " + answer.strip() + "\n")
    prompt.append("Is the possible answer:\n")
    prompt.append("A) True\n")
    prompt.append("B) False\n")
    prompt.append("The possible answer is: ")
    if is_correct is not None:
        prompt.append("A" if is_correct else "B")
    return "".join(prompt)


def compute_p_true(
        dataset: EmbeddingDataset, *,
        start_idx: int = 0,
        few_shot_num: int = 10,
        batch_size: int = 16,
) -> None:
    """
    Calculate the P-true for each answer in the list of answers.
    """
    mt = ModelAndTokenizer(
        config.MODEL_NAME,
        low_cpu_mem_usage=True,
    )
    base_prompt = ""
    for i in range(few_shot_num):
        is_correct = dataset[i]["accuracy"] == 1.0
        base_prompt += construct_p_true_prompt(dataset[i]["question"],
                                               dataset[i]["generated_text"],
                                               [ans['text'] for ans in dataset[i]["full_answers"]],
                                               is_correct=is_correct)
        base_prompt += "\n\n"

    if len(base_prompt) <= 2000:
        batch_size = 8
    elif len(base_prompt) <= 4000:
        batch_size = 2
    else:
        batch_size = 1

    prompts = []
    for idx in range(start_idx, len(dataset)):
        question = dataset[idx]["question"]
        answer = dataset[idx]["generated_text"]
        full_answers = [ans['text'] for ans in dataset[idx]["full_answers"]]
        prompts.append(base_prompt + construct_p_true_prompt(question, answer, full_answers,
                                                             True))

    logger.info("Generating P-true responses...")

    loss_fct = CrossEntropyLoss(reduction='none')

    for idx in tqdm(range(start_idx, len(dataset), batch_size), desc="Computing P-true"):

        if (idx // batch_size) % 100 == 0 and idx > start_idx:
            save_ckpt(dataset, f"full_result", idx)
            gc.collect()
            torch.cuda.empty_cache()

        input_data_list = prompts[idx:idx + batch_size]
        if len(input_data_list) == 0:
            break

        # Tokenize the batch of inputs
        tokenized_prompts = mt.tokenizer(input_data_list, return_tensors='pt', padding=True, truncation=True,
                                         max_length=8192
                                         ).to('cuda')
        input_ids = tokenized_prompts['input_ids']

        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        with torch.inference_mode():
            model_outputs = mt.model(input_ids, labels=target_ids)
            logits = model_outputs.logits

            # Compute per-token loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Reshape to (batch_size, seq_len)
        loss_per_token = loss_per_token.view(shift_labels.size())

        # Compute loss per sample by summing over sequence length
        loss_per_sample = loss_per_token.sum(dim=1)

        for i, loss in enumerate(loss_per_sample):
            # Compute probabilities
            dataset[idx + i]["p_true"] = -loss.item()
            dataset[idx + i]["p_true_fixed"] = torch.exp(-loss).item()
            dataset[idx + i]["p_false"] = 1 - dataset[idx + i]["p_true"]
            dataset[idx + i]["p_false_fixed"] = 1 - dataset[idx + i]["p_true_fixed"]

        del model_outputs

    del mt
    gc.collect()
    torch.cuda.empty_cache()


def construct_accuracy_prompt(
        question: str,
        reference: str,
        generated_answer: str,
) -> str:
    """
    Construct the prompt for accuracy uncertainty metric.
    :param question:
    :param reference: 标准答案
    :param generated_answer: 单次生成的答案，temperature=0.1
    :return:
    """
    prompt = f'We are assessing the quality of answers to the following question: {question}\n'
    prompt += f"The expected answer is: {reference}.\n"
    prompt += f"The proposed answer is: {generated_answer}\n"
    prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    prompt += " Respond only with yes or no.\nResponse:"
    return prompt


def compute_accuracy(
        model_name_or_path,
        dataset: EmbeddingDataset, *,
        start_idx: int = 0,
        max_length: int = 512,
        temperature: float = 0.01,
        tensor_parallel_size: int = 1,
        stop_sequences: list = None,
) -> None:
    """
    Calculate the accuracy for each answer in the list of answers.
    """

    if stop_sequences is None:
        stop_sequences = ["\n", "."]

    llm = LLM(model=model_name_or_path, max_model_len=2048, tensor_parallel_size=tensor_parallel_size,
              trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_length,
        stop=stop_sequences,
    )

    prompts = []
    for idx in range(start_idx, len(dataset)):
        question = dataset[idx]["question"]
        reference = dataset[idx]["ground_truth"]
        generated_answer = dataset[idx]["generated_text"]
        prompts.append(construct_accuracy_prompt(question, reference, generated_answer))

    logger.info("Generating accuracy responses...")

    responses = llm.generate(prompts, sampling_params=sampling_params)

    error_prompts = []
    error_idx = []
    for response in responses:
        vllm_resp_prompt = response.prompt
        out_idx = prompts.index(vllm_resp_prompt)

        if 'yes' in response.outputs[0].text.lower():
            dataset[out_idx + start_idx]["accuracy"] = 1.0
        elif 'no' in response.outputs[0].text.lower():
            dataset[out_idx + start_idx]["accuracy"] = 0.0
        else:
            error_prompts.append(vllm_resp_prompt)
            error_idx.append(out_idx + start_idx)

    if len(error_prompts) > 0:
        sampling_params_fallback = SamplingParams(
            temperature=1.0,
            max_tokens=max_length,
            stop=stop_sequences,
        )
        logger.info("Processing error responses, fallback...")
        responses_fallback = llm.generate(error_prompts, sampling_params=sampling_params_fallback)
        for response in responses_fallback:
            vllm_resp_prompt = response.prompt
            out_idx = error_prompts.index(vllm_resp_prompt)
            if 'yes' in response.outputs[0].text.lower():
                dataset[error_idx[out_idx]]["accuracy"] = 1.0
            elif 'no' in response.outputs[0].text.lower():
                dataset[error_idx[out_idx]]["accuracy"] = 0.0
            else:
                dataset[error_idx[out_idx]]["accuracy"] = 0.0

    del llm
    gc.collect()
    torch.cuda.empty_cache()
