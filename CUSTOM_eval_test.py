import os
import re
import argparse
import json
from components.utils import dump_json


OUTPUT_DIR = "test_results/"


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
    model_type = 'LLaMA2-7b' if dataset == 'WebQSP' else 'LLaMA-2-13b'
    epoch_cnt = '100' if dataset == 'WebQSP' else '10'
    DATA_FILE_PREDICT = f"Reading/{model_type}/{dataset}_Freebase_NQ_lora_epoch{epoch_cnt}/evaluation_beam/generated_predictions.jsonl"

    print('Loading data from:', DATA_FILE_PREDICT)
    with open(DATA_FILE_PREDICT, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


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
        if content.count(',') >= 2:
            result += 'rel'
        # [ ] --> entity
        else:
            result += 'ent'
        
        parse_idx = c_bracket_idx + 1
    
    return result


def same_rel_cnt(object: dict) -> bool:
    pred_r = object['pred_r']
    pred_s = object['pred_s']
    return pred_r == pred_s.count('rel')


def check_structure(dataloader: list, log_result: bool):
    print()
    print('Checking structure mismatches ')

    match_cnt = 0
    mismatch_cnt = 0
    total_cnt = 0
    match_data = []
    mismatch_data = []
    log_data = []

    for i, pred in enumerate(dataloader):
        predictions = pred['predict']   # list of rel_cnt/S-exp strings
        gold_label = pred['label']       # rel_cnt/S-exp string
        
        # split relation count & logical form
        pred_list = []
        for x in predictions:
            pred_split = re.split("{|}", x)
            pred_list.append([
                int(pred_split[1]), 
                pred_split[3][1:-1]
            ])
            
        gold_split = re.split("{|}", gold_label)
        gold_list = [int(gold_split[1]), gold_split[3][1:-1]]
        
        if gold_list[1].lower() == 'null':
            continue

        # remove entity/relation placeholder tokens
        for pred_rel, pred_lf in pred_list:
            total_cnt += 1
            pred_skeleton = remove_entity_relation_placeholders(pred_lf)
            gold_skeleton = remove_entity_relation_placeholders(gold_list[1])
            
            if pred_skeleton == gold_skeleton:
                match_cnt += 1
                match_data.append({
                    'NLQues': pred['input'][12:-1],
                    'pred_r': pred_rel,
                    'gold_r': gold_list[0],
                    'pred_s': pred_skeleton, 
                    'gold_s': gold_skeleton, 
                })
            else:
                mismatch_cnt += 1
                mismatch_obj = { 
                    'NLQues': pred['input'][12:-1],
                    'pred_r': pred_rel,
                    'gold_r': gold_list[0],
                    'pred_s': pred_skeleton, 
                    'gold_s': gold_skeleton, 
                    'pred_l': pred_lf,
                    'gold_l': gold_list[1],
                }
                mismatch_data.append(mismatch_obj)
    
                # Compare rel predict & rel included in LF
                if log_result and not same_rel_cnt(mismatch_obj):
                    log_data.append(mismatch_obj)
    
            # if not same_logical_form(pred_skeleton, gold_skeleton):
            #     output_data.append({ 'pred_s': pred_skeleton, 'gold_s': gold_skeleton, })
            
    # JSONL
    # jsonl_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_mismatch.jsonl')
    
    # with open(jsonl_file_path, 'w') as f:
    #     for item in mismatch_data:
    #         json_string = json.dumps(item)
    #         f.write(json_string + '\n')

    # print statistics
    print("Total predictions:", total_cnt)
    print("Match rate:", match_cnt / total_cnt)
    print("Mismatch rate:", mismatch_cnt / total_cnt)
    print()

    # JSON
    match_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_match.json')
    dump_json(match_data, match_file_path, indent=4)

    mismatch_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_mismatch.json')
    dump_json(mismatch_data, mismatch_file_path, indent=4)
    
    log_file_path = open_write_file(OUTPUT_DIR, f'log_lf_skeleton_mismatch_rel.json')
    dump_json(log_data, log_file_path, indent=4)


if __name__=='__main__':
    args = _parse_args()
    
    dataloader = prepare_dataloader(args.dataset)
    check_structure(dataloader, args.log)
