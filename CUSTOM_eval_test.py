import os
import re
import json
from components.utils import dump_json


DATA_FILE_PREDICT = "Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl"
OUTPUT_DIR = "test_results/"


def open_write_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def prepare_dataloader():
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
        if ',' in content:
            result += 'rel'
        # [ ] --> entity
        else:
            result += 'ent'
        
        parse_idx = c_bracket_idx + 1
    
    return result


def check_structure(dataloader: list):
    print()
    print('Checking structure mismatches ')

    match_data = []
    mismatch_data = []

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
            pred_skeleton = remove_entity_relation_placeholders(pred_lf)
            gold_skeleton = remove_entity_relation_placeholders(gold_list[1])
            
            if pred_skeleton == gold_skeleton:
                match_data.append({
                    'pred_r': pred_rel,
                    'gold_r': gold_list[0],
                    'pred_s': pred_skeleton, 
                    'gold_s': gold_skeleton, 
                })
            else:
                mismatch_data.append({ 
                    'pred_r': pred_rel,
                    'gold_r': gold_list[0],
                    'pred_s': pred_skeleton, 
                    'gold_s': gold_skeleton, 
                })

            # if not same_logical_form(pred_skeleton, gold_skeleton):
            #     output_data.append({ 'pred_s': pred_skeleton, 'gold_s': gold_skeleton, })
    
    # JSONL
    # jsonl_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_mismatch.jsonl')
    
    # with open(jsonl_file_path, 'w') as f:
    #     for item in mismatch_data:
    #         json_string = json.dumps(item)
    #         f.write(json_string + '\n')

    # JSON
    match_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_match.json')
    dump_json(match_data, match_file_path, indent=4)

    mismatch_file_path = open_write_file(OUTPUT_DIR, f'lf_skeleton_mismatch.json')
    dump_json(mismatch_data, mismatch_file_path, indent=4)


if __name__=='__main__':
    dataloader = prepare_dataloader()
    check_structure(dataloader)
