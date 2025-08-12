import os
import json
from components.utils import dump_json
from executor.logic_form_util import same_logical_form


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
        predictions = pred['predict']   # list of S-exp strings
        gen_label = pred['label']       # S-exp string

        if gen_label.lower() == 'null':
            continue

        # remove entity/relation placeholder tokens
        for predict in predictions:
            pred_skeleton = remove_entity_relation_placeholders(predict)
            gold_skeleton = remove_entity_relation_placeholders(gen_label)
            
            if pred_skeleton == gold_skeleton:
                match_data.append({ 'pred_s': pred_skeleton, 'gold_s': gold_skeleton, })
            else:
                mismatch_data.append({ 'pred_s': pred_skeleton, 'gold_s': gold_skeleton, })

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
