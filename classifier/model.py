import classifier.model_args as ma
from components.utils import load_json, dump_json

import os
import numpy as np
import evaluate
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from huggingface_hub import HfApi


OUTPUT_DIR = f"{ma.output_dir}/evaluation_beam"
CHECKPOINT_DIR = f"{ma.output_dir}/{ma.checkpoint_dir}"
LOG_DIR = f"{ma.log_dir}"


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def create_repo_clone():
    # create folder
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # huggingface API
    api = HfApi(token=ma.hf_auth_token_w)
    api.upload_folder(
        folder_path=CHECKPOINT_DIR,
        repo_id="raymonddasushi/chatkbqa-bert-classifier",
        repo_type="model",
    )


def load_classifier_model_and_tokenizer():
    config_kwargs = {
        "trust_remote_code": ma.trust_remote_code,
        "use_auth_token": ma.use_auth_token,
        "token": ma.hf_auth_token_w,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        ma.model_path,
        # use_fast=use_fast_tokenizer,
        # padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForSequenceClassification.from_pretrained(
        ma.model_path,
        num_labels=ma.num_labels,
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
 
    def tokenize(examples):
        return tokenizer(examples["question"], padding="max_length", truncation=True)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        # predictions += 1
        # labels += 1
        return metric.compute(predictions=predictions, references=labels)
    
    # make sure repo is cloned
    # create_repo_clone()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        
        push_to_hub=ma.push_to_hub,
        hub_model_id=ma.model_final_path,
        hub_private_repo=ma.hub_private_repo,
        
        # evaluation_strategy="steps",
        learning_rate=ma.learning_rate,
        per_device_train_batch_size=ma.per_device_train_batch_size,
        per_device_eval_batch_size=ma.per_device_eval_batch_size,
        gradient_accumulation_steps=ma.gradient_accumulation_steps,
        num_train_epochs=ma.num_train_epochs,
        weight_decay=ma.weight_decay,
        
        # logging_dir=LOG_DIR,
        # logging_steps=ma.logging_steps,
        
        report_to="none",   # wandb output disable
        save_strategy="steps",
        save_steps=ma.save_steps,
        resume_from_checkpoint=ma.checkpoint_path,
        load_best_model_at_end=ma.load_best_model_at_end,
        # no_cuda=True,       # use CPU
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.create_model_card(
        model_name=ma.model_final_name,
        finetuned_from=ma.model_path,
    )
    
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    trainer.save_state()
    # trainer.save_model()

    # trainer.push_to_hub(token=ma.hf_auth_token_w)


def load_and_run_classifier(dataset_name: str):
    # model_name_or_path = ma.model_final_path
    local_model_path = CHECKPOINT_DIR
    device = 0
    
    # data_train_path = f'data/{dataset_name}/generation/merged/{dataset_name}_train_class.json'
    data_test_path = f'data/{dataset_name}/generation/merged/{dataset_name}_test_class.json'
    # dataset = load_dataset("json", data_files={'train': data_train_path, 'test': data_test_path})
    dataset = load_json(data_test_path)
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    
    # adjust zero-base string labels to one-base integer labels
    model.config.id2label = {0:"1", 1:"2", 2:'3', 3:"4", 4:"5"}
    # model.config.label2id = {"1":1, '2':2, "3":3, "4":4, '5':5}
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, framework="pt")
    
    # inference
    total_cnt = 0
    match_cnt = 0
    predictions = []
    
    for item in tqdm(dataset):
        total_cnt += 1
        output = classifier(item['question'])
        pred = int(output[0]['label'])
        
        predictions.append({
            "question": item['question'],
            "gen_label": item['label'] + 1,     # change zero-index in dataset to one-index
            "predictions": pred,
        })
        
        if pred == item['label'] + 1:
            match_cnt += 1
    
    # print statistics
    print(f'Total lines: {total_cnt}')
    print(f'Matched lines: {match_cnt}')
    print(f'Percentage of matched lines: {match_cnt / total_cnt * 100}%')
    
    test_stats = {
        "total": total_cnt,
        "exmatch_num": match_cnt,
        "exmatch_rate": match_cnt / total_cnt,
    }
    output_stats_dir = open_write_file(OUTPUT_DIR, 'test_gen_statistics.json')
    dump_json(test_stats, output_stats_dir, indent=4)

    # print results
    output_results_dir = open_write_file(OUTPUT_DIR, 'generated_predictions.json')
    dump_json(predictions, output_results_dir, indent=4)

    
    # text = "what money does spain use"
    # output = classifier(text)
    # print(text)
    # print(output)