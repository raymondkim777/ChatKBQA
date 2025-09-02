from classifier import ClassifierModel
from llmtuner import ChatModel
import json
from tqdm import tqdm
import re
import os
# from llmtuner.tuner.core import get_infer_args


DATA_PIPELINE_PATH = os.path.join('LLMs/data', 'WebQSP_Freebase_NQ_test_pipeline', 'examples.json')
OUTPUT_DIR = os.path.join('Reading', 'Full_Pipeline', 'WebQSP_Freebase_NQ', 'evaluation_beam')


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def main():
    # model_args, data_args, _, _ = get_infer_args()
    class_model = ClassifierModel()
    chat_model = ChatModel()
    output_data = []
    
    with open(DATA_PIPELINE_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)        
        # random.shuffle(json_data)
        total_lines = 0
        matched_lines = 0
        will_matched_lines = 0

        for data in tqdm(json_data):
            total_lines += 1
            
            # classifer
            predicted_rel_cnt = class_model.classify(data['question'])  # integer
            
            # chatkbqa llm
            chat_input = 'Question: { ' + data['question'] + ' }, Relation Count: { ' + str(predicted_rel_cnt) + ' }'    
            query = data['chat_instruction'] + chat_input
            predict = chat_model.chat_beam(query)
            predict = [p[0] for p in predict]
            
            # output data
            output_data.append({
                'question': data['question'],
                'rel_label': data['rel_cnt'],
                'rel_predict': predicted_rel_cnt,
                'label': data['chat_output'],
                'predict': predict,
            })
            
            # statistics
            for p in predict:
                if data['chat_output'] == p:
                    matched_lines += 1
                    break
            for p in predict:
                if re.sub(r'\[.*?\]', '', data['chat_output']) == re.sub(r'\[.*?\]', '', p):
                    will_matched_lines += 1
                    break

    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")
    print(f"Will Matched lines: {will_matched_lines}")

    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")

    will_percentage = (will_matched_lines / total_lines) * 100
    print(f"Percentage of will matched lines: {will_percentage:.2f}%")
    
    # output JSON
    output_path = open_write_file(OUTPUT_DIR, 'generated_predictions.json')
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
