import json
import os
import re
from transformers import BertTokenizer, BertModel
from transformers.models.bert import WordpieceTokenizer
from scipy.spatial.distance import cosine


def not_empty(s):
    return s and s.strip()


def like_camel_to_tokens(camel_format):
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format


def read_dict_line(path):
    dict_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        data = json.loads(line)
        dict_list.append(data)
    return dict_list


def get_bert_embeddings(dataset_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/', do_lower_case=True, cache_dir='data/cache')

    bert_model = BertModel.from_pretrained(
        'bert-base-uncased/', output_hidden_states=True, cache_dir='data/cache'
    )

    service_operation_set = set()
    for file in os.listdir('data/%s/raw' % dataset_name):
        traces = read_dict_line(os.path.join('data/%s/raw' % dataset_name, file))
        for trace in traces:
            for span in trace['nodes']:
                service_operation_set.add(span['service'] + '/' + span['operation'])

    vector_map = {}
    vector_list = []
    print("get bert embedding...")

    for raw_text in service_operation_set:
        text1 = re.split(r'[-{},\!:=\[\]\(\)\$\s\.\/\#\|\\]', raw_text)
        text_list = []
        for token in text1:
            token = like_camel_to_tokens(token)
            text_list += token
        text_processed = " ".join(text_list)
        word_pieced = tokenizer.wordpiece_tokenizer.tokenize(text_processed)

        tokenized_text = tokenizer(text_processed, padding='max_length', max_length=50, return_tensors="pt")
        outputs = bert_model(**tokenized_text)
        results = outputs.pooler_output.tolist()[0]
        vector_map[raw_text] = results
        vector_list.append(results)
    with open("data/%s/preprocessed/bert_embeddings.json" % dataset_name, "w+") as f:
        f.write(json.dumps(vector_map))
