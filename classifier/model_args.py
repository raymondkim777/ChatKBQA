import os

# model metadata
model_final_name = "chatkbqa-bert-classifier"
model_final_path = f"raymonddasushi/{model_final_name}"
model_path = "google-bert/bert-base-uncased"
use_fast_tokenizer = False

# directory
dataset = "WebQSP"
output_dir = f"Reading/Bert-base-uncased/{dataset}_Freebase_NQ"
checkpoint_dir = f"checkpoint"
checkpoint_path = os.path.join(output_dir, checkpoint_dir)
__checkpoint_empty = False

if os.path.exists(checkpoint_path):
    __checkpoint_file_names = [name for name in os.listdir(checkpoint_path) if name.startswith('checkpoint')] 
    if len(__checkpoint_file_names) == 0:
        __checkpoint_empty = True
    else:
        __checkpoint_max_num = max([int(file_name[11:]) for file_name in __checkpoint_file_names])
        __checkpoint_name = f"checkpoint-{__checkpoint_max_num}"

        checkpoint_path = os.path.join(output_dir, checkpoint_dir, __checkpoint_name)
else:
    __checkpoint_empty = True
    
if __checkpoint_empty:
    checkpoint_path = False
print("Latest Checkpoint:", checkpoint_path)

log_dir = f"train_Bert-based-uncased_{dataset}_Freebase_NQ_log"

# hf parameters
trust_remote_code = True
use_auth_token = True
hf_auth_token = os.getenv("HF_AUTH_TOKEN")
hf_auth_token_w = os.getenv("HF_AUTH_TOKEN_W")

# model parameters
num_labels = 5
learning_rate = 3e-5
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 30
weight_decay = 0.1
# logging_steps = 10
save_steps = 1000
