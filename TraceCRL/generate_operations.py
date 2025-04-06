import json


def get_operations(dataset_name):
    for dataset_type in ['train', 'test']:
        with open('data/%s/%s/preprocessed/%s.json' % (dataset_name, dataset_type, dataset_type), 'r') as f:
            data = json.load(f)
        res = {}
        for k in data.keys():
            trace = data[k]
            for from_id, to_list in trace['edges'].items():
                for to in to_list:
                    if from_id == '0':
                        try:
                            api_pair = 'root--->' + '%s/%s' % (
                                trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1])
                            # api_pair = 'root--->' + trace['vertexs'][str(to['vertexId'])][1].replace(
                            #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                        except Exception as e:
                            print(e)
                            print(k)
                            print(trace)
                    else:
                        api_pair = ('%s/%s' % (trace['vertexs'][from_id][0], trace['vertexs'][from_id][1]) + '--->' +
                                    '%s/%s' % (
                                        trace['vertexs'][str(to['vertexId'])][0], trace['vertexs'][str(to['vertexId'])][1]))
                        # api_pair = trace['vertexs'][from_id][1].replace(trace['vertexs'][from_id][0] + '/', '') + \
                        #            '--->' + trace['vertexs'][str(to['vertexId'])][1].replace(
                        #     trace['vertexs'][str(to['vertexId'])][0] + '/', '')
                    if api_pair not in res.keys():
                        res[api_pair] = {'duration': []}
                    for i in res[api_pair].keys():
                        if i in to.keys():
                            res[api_pair][i].append(to[i])
                        else:
                            res[api_pair][i].append(0)
        with open('data/%s/%s/preprocessed/operations.json' % (dataset_name, dataset_type), 'w') as file:
            json.dump(res, file)
