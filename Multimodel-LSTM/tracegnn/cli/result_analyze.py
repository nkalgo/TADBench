from pprint import pprint

from tracegnn.utils import *
from tracegnn.visualize import *


@click.group()
def main():
    pass


@main.command()
@click.option('--nll-key', default='nll_list', required=True)
@click.option('--label-key', default='label_list', required=True)
@click.option('--up-sample-normal', type=int, default=None, required=False)
@click.option('--gui', is_flag=True, default=False, required=False)
@click.option('--proba-out', default=None, required=False)
@click.option('--auc-out', default=None, required=False)
@click.argument('input-file')
def analyze_nll(nll_key, label_key, up_sample_normal,
                gui, proba_out, auc_out, input_file):
    # check parameters
    if gui:
        proba_out = ':show:'
        auc_out = ':show:'

    # load file
    f = np.load(input_file)
    nll_list = f[nll_key]
    label_list = f[label_key]

    # up sample normal nll & label if required
    result_dict = analyze_anomaly_nll(
        nll_list=nll_list,
        label_list=label_list,
        up_sample_normal=up_sample_normal,
        proba_cdf_file=proba_out,
        auc_curve_file=auc_out,
    )
    pprint(result_dict)


if __name__ == '__main__':
    main()
