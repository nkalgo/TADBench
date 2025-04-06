import json
from collections import OrderedDict
import click

@click.command()
@click.option('--dataset_name', required=True, type=str)
def main(dataset_name):
    with open('../results.json', 'r') as f:
        data = json.load(f)

    algorithms = []
    for algo_name, metrics in data[dataset_name].items():
        algo_data = OrderedDict()
        algo_data['name'] = algo_name

        for metric_key in ['total', 'structure', 'latency']:
            metric_data = metrics.get(metric_key, {})
            converted = {
                'precision': metric_data.get('p', 0),
                'recall': metric_data.get('r', 0),
                'f1': metric_data.get('f1', 0),
                'accuracy': metric_data.get('acc', 0),
                'time': metric_data.get('time', 0)
            }
            algo_data[metric_key] = converted
    
        algorithms.append(algo_data)

    print(algorithms)

    with open('template.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    js_data = json.dumps(algorithms, indent=4, ensure_ascii=False)

    html = html.replace(
        'const algorithms = [/* DATA_PLACEHOLDER */]',
        f'const algorithms = {js_data}'
    )

    html = html.replace(
        'Benchmark on <span class="highlight"> </span>',
        f'Benchmark on <span class="highlight"> {dataset_name} </span>'
    )


    with open('leaderboard.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("HTML生成成功！请打开 leaderboard.html 查看结果")


if __name__ == '__main__':
    main()