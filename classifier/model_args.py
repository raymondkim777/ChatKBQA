import os

# model metadata
model_final_name = "Bert-based-uncased-ChatKBQA-classifier-rel-cnt"
model_final_path = f"raymonddasushi/{model_final_name}"
model_path = "google-bert/bert-base-uncased"
use_fast_tokenizer = False

# directory
dataset = "WebQSP"
output_dir = f"Reading/Bert-base-uncased/{dataset}_Freebase_NQ"
checkpoint_dir = f"checkpoint"
log_dir = f"train_Bert-based-uncased_{dataset}_Freebase_NQ_log"

# hf parameters
trust_remote_code = True
use_auth_token = True
hf_auth_token = os.getenv("HF_AUTH_TOKEN")

# model parameters
num_labels = 5
learning_rate = 3e-5
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 30
weight_decay = 0.1
logging_steps = 10
save_steps = 1000
