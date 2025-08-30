import os
from .model_args import model_path, use_fast_tokenizer

import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback, 
    Trainer,
)


OUTPUT_DIR = f"Reading/Bert-base-uncased/{model_path}_Freebase_NQ_lora/evaluation_beam"


def load_classifier_model_and_tokenizer():
    config_kwargs = {
        "trust_remote_code": True,
        "use_auth_token": True,
        "token": os.getenv("HF_AUTH_TOKEN"),
    }

    config = AutoConfig.from_pretrained(
        model_path,
        **config_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # use_fast=use_fast_tokenizer,
        # padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5,
        # torch_dtype=model_args.compute_dtype,
        # low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # # Push the config to your namespace with the name "chatkbqa-classifier-bert".
    # config.push_to_hub("chatkbqa-classifier-bert")

    # # Push the config to an organization with the name "my-finetuned-bert".
    # config.push_to_hub("huggingface/my-finetuned-bert")
    
    
    
    # from llmtuner.tuner.core.adapter import init_adapter
    # from llmtuner.tuner.core.utils import prepare_model_for_training

    # Initialize adapters
    # if is_trainable:
    #     model = prepare_model_for_training(model, model_args.layernorm_dtype, finetuning_args.finetuning_type)
    # model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
    # model = model.train() if is_trainable else model.eval()
    
    
    # Prepare model for inference
    # if not is_trainable:
    #     model.requires_grad_(False) # fix all model params
    #     model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    # trainable_params, all_param = count_parameters(model)
    # logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
    #     trainable_params, all_param, 100 * trainable_params / all_param
    # ))

    return model, tokenizer

    
    
def classifier_sft(dataset_name: str):
    data_train_path = f'data/{dataset_name}/generation/merged/{dataset_name}_train_class.json'
    data_test_path = f'data/{dataset_name}/generation/merged/{dataset_name}_test_class.json'
    dataset = load_dataset("json", data_files={'train': data_train_path, 'test': data_test_path})
    
    model, tokenizer = load_classifier_model_and_tokenizer()
    
    def encode(examples):
        return tokenizer(examples["question"], padding="max_length", truncation=True)
    
    dataset = dataset.map(encode, batched=True)
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    trainer.train()