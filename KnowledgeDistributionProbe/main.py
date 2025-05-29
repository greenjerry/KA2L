import gc
import os
import sys

import config
from baseline import compute_accuracy, compute_p_true
from utils import args_util, log_util
from utils.log_util import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle

import torch

from dataset import ChoiceDataset, QADataset, QADatasetWithContext, TriviaQADataset, NQOpenDataset, MedMCQADataset
from model import decode_tokens, make_inputs, ModelAndTokenizer, predict_from_input, \
    generate_first_with_hidden_states, generate_multiple_vllm
from entropy import compute_semantic_entropy

torch.set_grad_enabled(False)


def generate_answer(dataset):
    if os.path.exists(os.path.join(config.SAVE_PATH, "results", "hidden-states.pkl")):
        logger.info("Loading hidden-states.pkl from cache")
        with open(os.path.join(config.SAVE_PATH, "results", "hidden-states.pkl"), "rb") as f:
            embedding_dataset = pickle.load(f)
            embedding_dataset.del_hidden_states()

    else:
        if config.RESUME_FROM_CHECKPOINT and "/checkpoints/generate_first_with_hidden_states_" in config.RESUME_FROM_CHECKPOINT:
            logger.info(f"Loading {config.RESUME_FROM_CHECKPOINT} from checkpoint")
            with open(config.RESUME_FROM_CHECKPOINT, "rb") as f:
                embedding_dataset = pickle.load(f)
            ckpt_file_name = os.path.split(config.RESUME_FROM_CHECKPOINT)[-1].split(".")[0]
            start_idx = int(ckpt_file_name.split("_")[-1])
        else:
            embedding_dataset = None
            start_idx = 0
        mt = ModelAndTokenizer(
            config.MODEL_NAME,
            low_cpu_mem_usage=True,
        )
        embedding_dataset = generate_first_with_hidden_states(mt,
                                                              dataset,
                                                              batch_size=config.BATCH_SIZE,
                                                              max_length=256,
                                                              temperature=0.1,
                                                              resume_from_checkpoint=embedding_dataset,
                                                              start_idx=start_idx
                                                              )
        with open(os.path.join(config.SAVE_PATH, "results", "hidden-states.pkl"), "wb") as f:
            pickle.dump(embedding_dataset, f)
        del mt
        embedding_dataset.del_hidden_states()

    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(os.path.join(config.SAVE_PATH, "results", "generation-result.pkl")):
        logger.info("Loading generation-result.pkl from cache")
        with open(os.path.join(config.SAVE_PATH, "results", "generation-result.pkl"), "rb") as f:
            embedding_dataset = pickle.load(f)
            embedding_dataset.del_hidden_states()
    else:
        if config.RESUME_FROM_CHECKPOINT and "/checkpoints/generate_multiple_vllm_" in config.RESUME_FROM_CHECKPOINT:
            logger.info(f"Loading {config.RESUME_FROM_CHECKPOINT} from checkpoint")
            with open(config.RESUME_FROM_CHECKPOINT, "rb") as f:
                vllm_result = pickle.load(f)
            ckpt_file_name = os.path.split(config.RESUME_FROM_CHECKPOINT)[-1].split(".")[0]
            finished_idx = int(ckpt_file_name.split("_")[-1])
        else:
            finished_idx = 0
            vllm_result = None

        full_answers = generate_multiple_vllm(
            config.MODEL_NAME,
            dataset,
            gen_times=10,
            max_length=256,
            temperature=1.0,
            resume_from_checkpoint=vllm_result,
            start_idx=finished_idx
        )
        for i in range(len(embedding_dataset)):
            qid = embedding_dataset[i]["id"]
            embedding_dataset[i]["full_answers"] = full_answers[qid]

        with open(os.path.join(config.SAVE_PATH, "results", "generation-result.pkl"), "wb") as f:
            pickle.dump(embedding_dataset, f)
        embedding_dataset.del_hidden_states()

    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(os.path.join(config.SAVE_PATH, "results", "entropy-result.pkl")):
        logger.info("Loading entropy-result.pkl from cache")
        with open(os.path.join(config.SAVE_PATH, "results", "entropy-result.pkl"), "rb") as f:
            embedding_dataset = pickle.load(f)
            embedding_dataset.del_hidden_states()
    else:
        if config.RESUME_FROM_CHECKPOINT and "/checkpoints/entropy_result_" in config.RESUME_FROM_CHECKPOINT:
            logger.info(f"Loading {config.RESUME_FROM_CHECKPOINT} from checkpoint")
            with open(config.RESUME_FROM_CHECKPOINT, "rb") as f:
                embedding_dataset = pickle.load(f)
            ckpt_file_name = os.path.split(config.RESUME_FROM_CHECKPOINT)[-1].split(".")[0]
            start_idx = int(ckpt_file_name.split("_")[-1])
        else:
            start_idx = 0

        compute_semantic_entropy(embedding_dataset, start_idx=start_idx)

        with open(os.path.join(config.SAVE_PATH, "results", "entropy-result.pkl"), "wb") as f:
            pickle.dump(embedding_dataset, f)

    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(os.path.join(config.SAVE_PATH, "results", "accuracy-result.pkl")):
        logger.info("Loading accuracy-result.pkl from cache")
        with open(os.path.join(config.SAVE_PATH, "results", "accuracy-result.pkl"), "rb") as f:
            embedding_dataset = pickle.load(f)
            embedding_dataset.del_hidden_states()
    else:
        if config.RESUME_FROM_CHECKPOINT and "/checkpoints/accuracy_result_" in config.RESUME_FROM_CHECKPOINT:
            logger.info(f"Loading {config.RESUME_FROM_CHECKPOINT} from checkpoint")
            with open(config.RESUME_FROM_CHECKPOINT, "rb") as f:
                embedding_dataset = pickle.load(f)
            ckpt_file_name = os.path.split(config.RESUME_FROM_CHECKPOINT)[-1].split(".")[0]
            start_idx = int(ckpt_file_name.split("_")[-1])
        else:
            start_idx = 0

        compute_accuracy(
            config.MODEL_NAME,
            embedding_dataset,
            start_idx=start_idx,
        )

        with open(os.path.join(config.SAVE_PATH, "results", "accuracy-result.pkl"), "wb") as f:
            pickle.dump(embedding_dataset, f)

    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(os.path.join(config.SAVE_PATH, "results", "full-result.pkl")):
        logger.info("Loading full-result.pkl from cache")
        with open(os.path.join(config.SAVE_PATH, "results", "full-result.pkl"), "rb") as f:
            embedding_dataset = pickle.load(f)
            embedding_dataset.del_hidden_states()
    else:
        if config.RESUME_FROM_CHECKPOINT and "/checkpoints/full_result_" in config.RESUME_FROM_CHECKPOINT:
            logger.info(f"Loading {config.RESUME_FROM_CHECKPOINT} from checkpoint")
            with open(config.RESUME_FROM_CHECKPOINT, "rb") as f:
                embedding_dataset = pickle.load(f)
            ckpt_file_name = os.path.split(config.RESUME_FROM_CHECKPOINT)[-1].split(".")[0]
            start_idx = int(ckpt_file_name.split("_")[-1])
        else:
            start_idx = 0

        compute_p_true(embedding_dataset, start_idx=start_idx, batch_size=8)

        with open(os.path.join(config.SAVE_PATH, "results", "full-result.pkl"), "wb") as f:
            pickle.dump(embedding_dataset, f)

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Finished all tasks!!!")


def calc_knowledge_edge(mt: ModelAndTokenizer, prompt: str, ):
    inp = make_inputs(mt.tokenizer, [prompt])
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])


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

    generate_answer(dataset)


if __name__ == "__main__":
    parser = args_util.get_parser()
    args, unknown = parser.parse_known_args()
    if args.config is not None:
        config.update_config(args.config)

    log_util.init_logger()
    logger.info('Starting new run with args: %s', args)

    main()
