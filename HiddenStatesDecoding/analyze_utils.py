import os
import shlex
import sys
from typing import Optional

import torch
import transformers
from accelerate.state import PartialState
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

import experiments
from .run_args import DataArguments, ModelArguments, TrainingArguments

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
transformers.logging.set_verbosity_error()

#############################################################################


def load_experiment_and_trainer(
    checkpoint_folder: str,
    args_str: Optional[str] = None,
    checkpoint: Optional[str] = None,
    do_eval: bool = True,
    sanity_decode: bool = True,
    max_seq_length: Optional[int] = None,
    use_less_data: Optional[int] = None,
):  # (can't import due to circluar import) -> trainers.InversionTrainer:
    # import previous aliases so that .bin that were saved prior to the
    # existence of the vec2text module will still work.

    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder)  # a checkpoint
    if checkpoint is None:
        # This happens in a weird case, where no model is saved to saves/xxx/checkpoint-*/pytorch_model.bin
        # because checkpointing never happened (likely a very short training run) but there is still a file
        # available in saves/xxx/pytorch_model.bin.
        checkpoint = checkpoint_folder
    print("[analyze_utils] Loading model from checkpoint:", checkpoint)


    cwd = os.path.dirname(
        os.path.abspath(__file__)
    )
    print(f"[analyze_utils] adding cwd to path: {cwd}")
    sys.path.append(cwd)

    if args_str is not None:
        args = shlex.split(args_str)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=args
        )
    else:
        print("[analyze_utils] loading args from checkpoint", checkpoint)
        try:
            print("[analyze_utils] loading data_args from", os.path.join(checkpoint, os.pardir, "data_args.bin"))
            data_args = torch.load(os.path.join(checkpoint, os.pardir, "data_args.bin"))
        except (FileNotFoundError):
            data_args = torch.load(os.path.join(checkpoint, "data_args.bin"))
        try:
            model_args = torch.load(
                os.path.join(checkpoint, os.pardir, "model_args.bin")
            )
        except (FileNotFoundError):
            model_args = torch.load(os.path.join(checkpoint, "model_args.bin"))
        try:
            training_args = torch.load(
                os.path.join(checkpoint, os.pardir, "training_args.bin")
            )
        except (FileNotFoundError):
            training_args = torch.load(os.path.join(checkpoint, "training_args.bin"))

    training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.use_wandb = False
    training_args.report_to = []
    training_args.mock_embedder = False
    training_args.no_cuda = not torch.cuda.is_available()

    if max_seq_length is not None:
        print(
            f"Overwriting max sequence length from {model_args.max_seq_length} to {max_seq_length}"
        )
        model_args.max_seq_length = max_seq_length

    if use_less_data is not None:
        print(
            f"Overwriting use_less_data from {data_args.use_less_data} to {use_less_data}"
        )
        data_args.use_less_data = use_less_data

    # For batch decoding outputs during evaluation.
    # os.environ["TOKENIZERS_PARALLELISM"] = "True"

    ########################################################################
    print("> checkpoint:", checkpoint)
    if (
        checkpoint
        == "/home/jxm3/research/retrieval/inversion/saves/47d9c149a8e827d0609abbeefdfd89ac/checkpoint-558000"
    ):
        # Special handling for one case of backwards compatibility:
        #   set dataset (which used to be empty) to nq
        data_args.dataset_name = "nq"
        print("set dataset to nq")

    if not torch.cuda.is_available():
        print("[analyze_utils] No GPU available, loading model on CPU")
        training_args.use_cpu = True
        training_args._n_gpu = 0
        training_args.local_rank = -1  # Don't load in DDP
        training_args.distributed_state = PartialState()
        training_args.deepspeed_plugin = None  # For backwards compatibility
        training_args.bf16 = 0  # no bf16 in case no support from GPU
    
    # Need to delete this cached property so that it's properly recomputed.
    if '__cached__setup_devices' in training_args.__dict__:
        del training_args.__dict__["__cached__setup_devices"]

    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer.model._keys_to_ignore_on_save = []
    try:
        trainer._load_from_checkpoint(checkpoint)
    except RuntimeError:
        # backwards compatibility from adding/removing layernorm
        trainer.model.use_ln = False
        trainer.model.layernorm = None
        # try again without trying to load layernorm
        trainer._load_from_checkpoint(checkpoint)
    if torch.cuda.is_available() and sanity_decode:
        trainer.sanity_decode()
    return experiment, trainer


def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args
