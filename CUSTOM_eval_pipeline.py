import os
import argparse
import json
from components.utils import load_json, dump_json


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def _parse_args():
    """Parse arguments: --dataset, --log"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebQSP', help='dataset to perform entity linking, should be WebQSP or CWQ')
    parser.add_argument('--log', action='store_true', help='outputs log in test_results/test_log.json')
    return parser.parse_args()


def prepare_dataloader(dataset: str):
    data_file_path = f'Reading/Full_Pipeline/{dataset}_Freebase_NQ/evaluation_beam/generated_predictions.json'
    return load_json(data_file_path)


def remove_entity_relation_placeholders(output: str):
    parse_idx = 0
    result = ''
    
    while parse_idx < len(output):
        if '[' not in output[parse_idx:]:
            result += output[parse_idx:]
            break
        try:
            o_bracket_idx = output.index('[', parse_idx)
            c_bracket_idx = output.index(']', parse_idx)
        except Exception:
            return output
        
        # found open bracket
        result += output[parse_idx: o_bracket_idx]
        content = output[o_bracket_idx: c_bracket_idx + 1]
        
        # [ , , ] --> relation
        if ',' in content:
            result += 'rel'
        # [ ] --> entity
        else:
            result += 'ent'
        
        parse_idx = c_bracket_idx + 1
    
    return result


def check_structure(dataloader: list, dataset: str, log_result: bool):
    print()
    print('Checking Classifier (rel_cnt prediction) & LLM (LF generation) Results')
    
    # data structure:
    # {
    #     "question": "what does jamaican people speak",
    #     "rel_label": 1,
    #     "rel_predict": 1,
    #     "label": "( JOIN ( R [ location , country , languages spoken ] ) [ Jamaica ] )",
    #     "predict": [
    #         "( JOIN ( R [ location , country , languages spoken ] ) [ Jamaica ] )"
    #     ]
    # },

    match_cnt = 0
    mismatch_cnt = 0
    total_cnt = 0
    rel_predict_match_cnt = 0
    rel_predict_mismatch_cnt = 0
    rel_predict_lf_mismatch_cnt = 0
    rel_match_lf_mismatch_cnt = 0
    rel_match_lf_rel_match_lf_mismatch_cnt = 0

    match_data = []
    mismatch_data = []
    log_rel_mismatch_freq = {1: 0, 2:0, 3:0, 4:0, 5:0}
    log_rel_mismatch_data = []
    log_rel_predict_lf_mismatch_data = []
    log_rel_match_lf_mismatch_data = []
    
    for pred in dataloader:
        predictions = pred['predict']   # list of S-exp strings
        gen_label = pred['label']       # S-exp string

        if gen_label.lower() == 'null':
            continue

        # remove entity/relation placeholder tokens
        for predict in predictions:
            total_cnt += 1
            pred_skeleton = remove_entity_relation_placeholders(predict)
            gold_skeleton = remove_entity_relation_placeholders(gen_label)
            
            if pred_skeleton == gold_skeleton:
                match_cnt += 1
                rel_predict_match_cnt += 1
                match_data.append({
                    'NLQuest': pred['question'],
                    'rel_cnt': pred['rel_label'],
                    'rel_pre': pred['rel_predict'],
                    'pred_sk': pred_skeleton, 
                    'gold_sk': gold_skeleton,
                })
            else:
                mismatch_cnt += 1
                mismatch_obj = {
                    'NLQuest': pred['question'],
                    'rel_cnt': pred['rel_label'],
                    'rel_pre': pred['rel_predict'],
                    'pred_sk': pred_skeleton, 
                    'gold_sk': gold_skeleton,
                    'pred_lf': predict,
                    'gold_lf': gen_label,
                }
                mismatch_data.append(mismatch_obj)
                
                # classifier performance
                if pred['rel_label'] != pred['rel_predict']:
                    rel_predict_mismatch_cnt += 1
                    if log_result:
                        log_rel_mismatch_freq[pred['rel_predict']] += 1
                        log_rel_mismatch_data.append(mismatch_obj)
                else:  # pipeline performance
                    rel_predict_match_cnt += 1
                    rel_match_lf_mismatch_cnt += 1
                    if log_result:
                        log_rel_match_lf_mismatch_data.append(mismatch_obj)
                    if pred_skeleton.count('rel') == pred['rel_predict']:
                        rel_match_lf_rel_match_lf_mismatch_cnt += 1
                
                # llm performance
                if pred_skeleton.count('rel') != pred['rel_predict']:
                    rel_predict_lf_mismatch_cnt += 1
                    if log_result:
                        log_rel_predict_lf_mismatch_data.append(mismatch_obj)

    # print statistics
    print("Total predictions:", total_cnt)
    print("Overall Match rate:", match_cnt / total_cnt)
    print("Overall Mismatch rate:", mismatch_cnt / total_cnt)
    print("Classifier Rel(X) rate:", rel_predict_mismatch_cnt / total_cnt)
    print("LLM Rel(X) rate:", rel_predict_lf_mismatch_cnt / total_cnt)
    print("Classifier Rel(O) LLM LF(X) rate:", rel_match_lf_mismatch_cnt / rel_predict_match_cnt)
    print("Classifier Rel(O) LLM Rel(O) LLM LF(X) rate:", rel_match_lf_rel_match_lf_mismatch_cnt / rel_predict_match_cnt)
    print()

    # JSON
    output_dir = f"Reading/Full_Pipeline/{dataset}_Freebase_NQ/test_results"
    
    match_file_path = open_write_file(output_dir, f'lf_skeleton_match.json')
    dump_json(match_data, match_file_path, indent=4)

    mismatch_file_path = open_write_file(output_dir, f'lf_skeleton_mismatch.json')
    dump_json(mismatch_data, mismatch_file_path, indent=4)
    
    log1_file_path = open_write_file(output_dir, f'log_rel_mismatch.json')
    dump_json(log_rel_mismatch_data, log1_file_path, indent=4)
    
    log2_file_path = open_write_file(output_dir, f'log_rel_match_lf_mismatch.json')
    dump_json(log_rel_match_lf_mismatch_data, log2_file_path, indent=4)


if __name__=='__main__':
    args = _parse_args()
    
    dataloader = prepare_dataloader(args.dataset)
    check_structure(dataloader, args.dataset, args.log)
