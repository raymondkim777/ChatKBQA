import argparse
from classifier.model import classifier_sft, load_and_run_classifier

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="WebQSP", type=str, help="CWQ | WebQSP")
    parser.add_argument('--do_train', action='store_true', help='run classifier sft')
    parser.add_argument('--do_infer', action='store_true', help='run inference on test dataset')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    if args.do_train:
        classifier_sft(args.dataset_type)
    if args.do_infer:
        load_and_run_classifier(args.dataset_type)
    

# CUDA_VISIBLE_DEVICES=3 nohup python -u CUSTOM_run_classifier.py >> train_Bert-based-uncased_WebQSP_Freebase_NQ_epoch50.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch --num_processes 2 --num_machines 1  CUSTOM_run_classifier.py >> train_Bert-baswed-uncased_WebQSP_Freebase_NQepoch100.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u CUSTOM_run_classifier.py >> predbeam_Bert-based-uncased_WebQSP_Freebase_NQ_epoch_50.txt 2>&1 &