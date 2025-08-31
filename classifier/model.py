import classifier.model_args as ma

import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


OUTPUT_DIR = f"{ma.output_dir}/evaluation_beam"
CHECKPOINT_DIR = f"{ma.output_dir}/{ma.checkpoint_dir}"
LOG_DIR = f"{ma.log_dir}"


def load_classifier_model_and_tokenizer():
    config_kwargs = {
        "trust_remote_code": ma.trust_remote_code,
        "use_auth_token": ma.use_auth_token,
        "token": ma.hf_auth_token,
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

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="epoch",
        
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
    trainer.save_model()

    trainer.push_to_hub("End of training", token=ma.hf_auth_token_w)


def load_classifier():
    model_name_or_path = ma.model_final_path
    device = "cuda"  # or "cpu" if no GPU
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # maybe ma.model_path?
    
    inputs = tokenizer.encode("what does jamaican people speak", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    
    print(type(outputs))
    print(outputs)