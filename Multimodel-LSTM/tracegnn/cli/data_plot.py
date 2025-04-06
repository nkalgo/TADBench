from tracegnn.data import *
from tracegnn.visualize import *


@click.group()
def main():
    pass


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-file', required=False, default=None)
@click.option('-n', '--graph-count', type=int, required=False, default=20)
@click.option('--no-browser', is_flag=True, default=False)
def plot_samples(input_dir, output_file, graph_count, no_browser):
    # check parameters
    if output_file is not None:
        _, ext = os.path.splitext(output_file)
        ext = ext.lower()
    else:
        ext = None

    # plot image
    if ext in ('.jpg', '.png'):
        db, id_manager = open_trace_graph_db(input_dir)
        fig = plot_grid(
            plot_trace_graph,
            db.sample_n(graph_count),
            id_manager=id_manager,
        )

        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()

    elif ext in (None, '.html', '.htm'):
        def fn():
            db, id_manager = open_trace_graph_db(input_dir)
            with db:
                return render_trace_graph_html(
                    db.sample_n(graph_count, with_id=True),
                    id_manager=id_manager,
                    output_file=output_file,
                    cdn=False,
                )

        if output_file is None:
            serve_html(fn, open_browser=not no_browser)
        else:
            fn()

    else:
        raise ValueError(f'Unsupported output format: {ext!r}')


if __name__ == '__main__':
    main()
