from classifier.model import classifier_sft

if __name__ == "__main__":
    classifier_sft('WebQSP')
    

# CUDA_VISIBLE_DEVICES=3 nohup python -u CUSTOM_classifier_train.py >> train_Bert-based-uncased_WebQSP_Freebase_NQepoch50.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch --num_processes 2 --num_machines 1  CUSTOM_classifier_train.py >> train_Bert-baswed-uncased_WebQSP_Freebase_NQepoch100.txt 2>&1 &
