from classifier.classifier_model import ClassifierModel
from llmtuner import ChatModel
import json
from tqdm import tqdm
import re
import os
from llmtuner.tuner.core import get_infer_args


DATA_PIPELINE_PATH = os.path.join('LLMs/data', 'WebQSP_Freebase_NQ_test_pipeline', 'examples.json')


def main():
    model_args, data_args, _, _ = get_infer_args()
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
            predicted_rel_cnt = class_model.classify(data['question'])
            
            # chatkbqa llm
            chat_input = 'Question: { ' + data['question'] + ' }, Relation Count: { ' + predicted_rel_cnt + ' }'    
            query = data['instruction'] + chat_input
            predict = chat_model.chat_beam(query)
            predict = [p[0] for p in predict]
            output_data.append({'label':data['output'],'predict':predict})
            for p in predict:
                if data['output'] == p:
                    matched_lines += 1
                    break
            for p in predict:
                if re.sub(r'\[.*?\]', '', data['output']) == re.sub(r'\[.*?\]', '', p):
                    will_matched_lines += 1
                    break
       

    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")
    print(f"Will Matched lines: {will_matched_lines}")

    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")

    will_percentage = (will_matched_lines / total_lines) * 100
    print(f"Percentage of will matched lines: {will_percentage:.2f}%")
    
    
    output_dir = os.path.join(os.path.dirname(model_args.checkpoint_dir[0]),'evaluation_beam/generated_predictions.jsonl')
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    with open(output_dir, 'w') as f:
        for item in output_data:
            json_string = json.dumps(item)
            f.write(json_string + '\n')
    
if __name__ == "__main__":
    main()
