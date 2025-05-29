import json
import os.path
from datetime import datetime

RUN_NAME = "default"
MODEL_NAME = "/data/LLM/Qwen2.5-7B"  # or "EleutherAI/gpt-j-6B" or
SAVE_PATH = "/home/airohit/yinhx//KnowledgeEdgeDetect"
DATASET_PATH = "/data/yinhx/dataset/agri_train_all_filtered.json"
LOG_LEVEL = "DEBUG"
ENTAILMENT_DEBERTA_PATH = "/data/LLM/deberta-v2-xlarge-mnli"
DATASET_TYPE = "qa"
DATASET_DATA_SAMPLE_SIZE = 100
BATCH_SIZE = 32
FEW_SHOT_NUM = 5
RESUME_FROM_CHECKPOINT = None
EMBEDDINGS_FROM_LAYER_N = -1


def update_config(json_path, continue_training=False):
    global MODEL_NAME, SAVE_PATH, DATASET_PATH, LOG_LEVEL, ENTAILMENT_DEBERTA_PATH, DATASET_TYPE, DATASET_DATA_SAMPLE_SIZE, RUN_NAME, BATCH_SIZE, FEW_SHOT_NUM, RESUME_FROM_CHECKPOINT, EMBEDDINGS_FROM_LAYER_N
    with open(json_path, "r") as f:
        config = json.load(f)
    RUN_NAME = config["run_name"]
    MODEL_NAME = config["model_name"]
    SAVE_PATH = config["save_path"]
    DATASET_PATH = config["dataset_path"]
    LOG_LEVEL = config["log_level"]
    ENTAILMENT_DEBERTA_PATH = config["entailment_deberta_path"]
    DATASET_TYPE = config["dataset_type"]
    DATASET_DATA_SAMPLE_SIZE = config["dataset_data_sample_size"]
    BATCH_SIZE = config.get("batch_size", 32)
    FEW_SHOT_NUM = config.get("few_shot_num", 5)
    RESUME_FROM_CHECKPOINT = config.get("resume_from_checkpoint", None)
    EMBEDDINGS_FROM_LAYER_N = config.get("embeddings_from_layer_n", -1)

    if RESUME_FROM_CHECKPOINT is None and not continue_training and os.path.exists(SAVE_PATH):
        prev_path, last_dirname = os.path.split(SAVE_PATH)
        new_dirname = last_dirname + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        SAVE_PATH = os.path.join(prev_path, new_dirname)
        print(f"Path {os.path.join(prev_path, last_dirname)} already exists. Saving to {SAVE_PATH} instead.")

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(os.path.join(SAVE_PATH, "checkpoints")):
        os.makedirs(os.path.join(SAVE_PATH, "checkpoints"))
    if not os.path.exists(os.path.join(SAVE_PATH, "logs")):
        os.makedirs(os.path.join(SAVE_PATH, "logs"))
    if not os.path.exists(os.path.join(SAVE_PATH, "results")):
        os.makedirs(os.path.join(SAVE_PATH, "results"))
    if not os.path.exists(os.path.join(SAVE_PATH, "cache")):
        os.makedirs(os.path.join(SAVE_PATH, "cache"))

    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(SAVE_PATH, f"run_config_{now_time}.json"), "w") as f:
        json.dump({
            "run_name": RUN_NAME,
            "model_name": MODEL_NAME,
            "save_path": SAVE_PATH,
            "dataset_path": DATASET_PATH,
            "log_level": LOG_LEVEL,
            "entailment_deberta_path": ENTAILMENT_DEBERTA_PATH,
            "dataset_type": DATASET_TYPE,
            "dataset_data_sample_size": DATASET_DATA_SAMPLE_SIZE
        }, f, indent=4, ensure_ascii=False)
