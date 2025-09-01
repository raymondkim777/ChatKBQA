import classifier.model_args as ma
from classifier.model import classifier_sft, load_and_run_classifier
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


CHECKPOINT_DIR = f"{ma.output_dir}/{ma.checkpoint_dir}"
DEVICE = 0


class ClassifierModel:
    
    def __init__(self):
        # model_name_or_path = ma.model_final_path
        local_model_path = CHECKPOINT_DIR
        
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        
        # adjust zero-base string labels to one-base integer labels
        model.config.id2label = {0:"1", 1:"2", 2:'3', 3:"4", 4:"5"}
        # model.config.label2id = {"1":1, '2':2, "3":3, "4":4, '5':5}
        
        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=DEVICE, framework="pt")
    
    
    def classify(self, query) -> int:
        # pipeline output: [{'label': '0', 'score': 0.9999728202819824}]
        return int(self.pipeline(query)['label'])