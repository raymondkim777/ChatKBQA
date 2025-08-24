from transformers import AutoConfig

config = AutoConfig.from_pretrained("google-bert/bert-base-cased")





# Push the config to your namespace with the name "my-finetuned-bert".
config.push_to_hub("my-finetuned-bert")

# Push the config to an organization with the name "my-finetuned-bert".
config.push_to_hub("huggingface/my-finetuned-bert")