import gc
import re

from torch import nn
from vllm import LLM, SamplingParams

from dataset import EmbeddingDataset
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from utils.save_utils import save_ckpt

STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:', 'Answer:', '问题：', '回答：', '上下文：',
                  '问题:',
                  '回答:',
                  '上下文:', ]


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
            self,
            model_name=None,
            model=None,
            tokenizer=None,
            low_cpu_mem_usage=True,
    ):
        if tokenizer is None:
            assert model_name is not None

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                      padding_side="left"
                                                      )
            if 'llama' in model_name.lower() or 'minicpm' in model_name.lower() or 'mistral' in model_name.lower():
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            # tokenizer.pad_token = "[PAD]"
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.bfloat16, trust_remote_code=True,
                device_map="auto"
            )
            model.eval()

        self.model_path_or_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
        self.stop_sequences = STOP_SEQUENCES + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name else 2048
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


class LlamaMLPClassifier(nn.Module):
    """
    https://github.com/ziweiji/Internal_States_Reveal_Hallucination/blob/main/classifier/classifier_models.py#L93

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, input_dim=4096, hidden_dim=14336):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.act_fn = nn.SiLU()
        self.score = nn.Linear(self.input_dim, 2)  # binary classification at this moment

    def forward(self, x):
        # x.shape # [b, num_layers, hidden_dim]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = self.score(down_proj)
        return x

    def get_device(self):
        return next(self.parameters()).device


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, tokenizer, inp):
    input = make_inputs(tokenizer, [inp])
    with torch.no_grad():
        out = model.generate(**input,
                             return_dict_in_generate=True,
                             output_scores=True,
                             temperature=1.0,

                             )
    answer = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    if answer.startswith(inp):
        input_data_offset = len(inp)
        answer = answer[input_data_offset:]
    return answer


def generate_first_with_hidden_states(
        mt: ModelAndTokenizer,
        dataset,
        max_input_length: int = 512,
        max_length: int = 256,
        batch_size: int = 16,
        temperature: float = 0.7,
        resume_from_checkpoint: EmbeddingDataset = None,
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
    if not resume_from_checkpoint:
        result: EmbeddingDataset = EmbeddingDataset()
    else:
        result = resume_from_checkpoint

    for idx in tqdm(range(start_idx, len(dataset), batch_size), desc="Sampling First with Hidden States"):
        if (idx // batch_size) % 100 == 0 and idx > start_idx:
            save_ckpt(result, f"generate_first_with_hidden_states", idx)
            gc.collect()
            torch.cuda.empty_cache()

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
                output_scores=True,
                temperature=temperature,
                # top_k=top_k,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=mt.tokenizer.pad_token_id,
            )

        transition_scores = mt.model.compute_transition_scores(
            model_out.sequences, model_out.scores, normalize_logits=True)
        log_likelihoods = [[score.item() for score in scores] for scores in transition_scores]

        hidden_states = model_out.hidden_states
        output_token_max_size = len(hidden_states)
        if model_out.sequences.size(1) != output_token_max_size + input_max_lengths:
            print(f"Warning: model_out.sequences.size(1)={model_out.sequences.size(1)}, "
                  f"output_token_max_size={output_token_max_size}, input_max_lengths={input_max_lengths}")
        output_texts = mt.tokenizer.batch_decode(model_out.sequences, skip_special_tokens=True)
        last_tok_bef_gen = hidden_states[0]

        last_tok_bef_gen_embedding = torch.stack([t[:, -1, :] for t in last_tok_bef_gen], dim=1)
        for i in range(current_batch_length):
            output_texts_tmp = output_texts[i][len(batch_prompts[i]):].split('\n')[0]
            log_likelihoods_tmp = log_likelihoods[i][:len(output_texts_tmp)]

            result.append(
                id=batch_dataset[i]["id"],
                prompt=batch_dataset[i]["prompt"],
                question=batch_dataset[i]["question"],
                generated_text=output_texts_tmp,
                full_answers=list(),
                hidden_last_tok_before_gen=last_tok_bef_gen_embedding[i].clone().cpu(),
                ground_truth=batch_dataset[i]["answer"],
                log_likelihood=log_likelihoods_tmp,
            )
        # del model_out, hidden_states, last_tok_bef_gen, last_tok_bef_gen_list, last_tok_bef_gen_embedding
        del model_out, hidden_states, last_tok_bef_gen, last_tok_bef_gen_embedding
        gc.collect()
        # torch.cuda.empty_cache()
    save_ckpt(result, f"generate_first_with_hidden_states", len(dataset))
    torch.cuda.empty_cache()
    return result


def generate_multiple_vllm(
        model_path: str,
        dataset,
        max_input_length: int = 512,
        gen_times: int = 10,
        max_length: int = 256,
        temperature: float = 1.0,
        tensor_parallel_size: int = 1,
        stop_sequences: list = None,
        resume_from_checkpoint: dict = None,
        start_idx: int = 0,
):
    """
    使用 vLLM 加速生成，并自动调整 batch size。

    Args:
        model_path:  Hugging Face 模型路径或本地模型目录。
        dataset:  数据集，每个元素包含 'id' 和 'prompt' 键。
        max_input_length:  最大输入长度。
        gen_times:  每个 prompt 生成的次数。
        max_length:  生成文本的最大长度（包括 prompt）。
        temperature:  采样温度。
        tensor_parallel_size:  张量并行大小（GPU 数量）。

    Returns:
        一个字典，键为 'id'，值为生成文本列表。
    """
    if stop_sequences is None:
        stop_sequences = STOP_SEQUENCES

    llm = LLM(model=model_path, max_model_len=2048, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_length,
        stop=stop_sequences,
        logprobs=0,
    )

    if resume_from_checkpoint is not None:
        result = resume_from_checkpoint
    else:
        result = {d['id']: [] for d in dataset}

    for epoch in tqdm(range(start_idx, gen_times), desc="Sampling Multiple Times with vLLM"):
        prompts = [d["prompt"] for d in dataset]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            prompt_index = prompts.index(prompt)

            data_id = dataset[prompt_index]['id']
            token_ids = output.outputs[0].token_ids
            logprobs = [probs[token_ids[idx]].logprob for idx, probs in
                        enumerate(output.outputs[0].logprobs)]
            result[data_id].append({
                "text": generated_text,
                "logprobs": logprobs,
                "token_ids": token_ids,
            })

        save_ckpt(result, f"generate_multiple_vllm", (epoch + 1))
        gc.collect()
        torch.cuda.empty_cache()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return result


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""

    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False
