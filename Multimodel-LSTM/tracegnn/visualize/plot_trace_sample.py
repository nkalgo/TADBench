from click import style
from graphviz import Digraph

def get_text_drop_anomaly(name, latency, times=1):
    return f"""<<FONT COLOR="tomato3" POINT-SIZE="9">×{times}</FONT>""" \
           f"""<BR ALIGN="RIGHT" /><FONT POINT-SIZE="16" COLOR="tomato3"><B>{name}</B></FONT>""" \
           f"""<BR/><FONT COLOR="tomato3" POINT-SIZE="12">{latency}ms</FONT>>"""

def get_text_latency_anomaly(name, latency, times=1):
    return f"""<<FONT COLOR="floralwhite" POINT-SIZE="9">×{times}</FONT>""" \
           f"""<BR ALIGN="RIGHT" /><FONT POINT-SIZE="16" COLOR="white"><B>{name}</B></FONT>""" \
           f"""<BR/><FONT COLOR="red" POINT-SIZE="12"><B>{latency}ms</B></FONT>>"""

def get_text(name, latency, times=1):
    return f"""<<FONT COLOR="floralwhite" POINT-SIZE="9">×{times}</FONT>""" \
           f"""<BR ALIGN="RIGHT" /><FONT POINT-SIZE="16" COLOR="white"><B>{name}</B></FONT>""" \
           f"""<BR/><FONT COLOR="white" POINT-SIZE="12">{latency}ms</FONT>>"""

def get_text_without_times(name, latency):
    return f"""<<FONT COLOR="floralwhite" POINT-SIZE="9"> </FONT>""" \
           f"""<BR ALIGN="RIGHT" /><FONT POINT-SIZE="16" COLOR="white"><B>{name}</B></FONT>""" \
           f"""<BR/><FONT COLOR="white" POINT-SIZE="12">{latency}ms</FONT>>"""


def plot_trace_sample_normal():
    G = Digraph(
        format='pdf',
        graph_attr={'size': '5.0,3.0!', 'nodesep': '1.0'},
        node_attr={'shape': 'rect', 'style': 'rounded,filled', 'width': '0.85', 'height': '0.7'},
        edge_attr={'arrowsize': '0.6', 'penwidth': '1.5'},
        engine='fdp'
    )

    G.node('Checkout', label=get_text('Checkout', 23), color='lightpink3', pos='1.1,0.6!')
    G.node('CheckPrice', label=get_text('CheckPrice', 5.5, 2), color='lightskyblue4', pos='0.75,0.3!')
    G.node('ReadDB', label=get_text('ReadDB', 2.5, 2), color='aquamarine4', pos='0.6,0!')
    G.node('WriteLog', label=get_text('WriteLog', 1), color='bisque4', pos='0.9,0.0!')
    G.node('Payment', label=get_text('Payment', 9), color='navajowhite3', pos='1.5,0.3!')
    G.node('ReadDB1', label=get_text('ReadDB', 2), color='aquamarine4', pos='1.2,0!')
    G.node('WriteDB', label=get_text('WriteDB', 4), color='mediumpurple3', pos='1.5,0!')
    G.node('WriteLog1', label=get_text('WriteLog', 1), color='bisque4', pos='1.8,0!')


    G.edge('Checkout', 'CheckPrice')
    G.edge('CheckPrice', 'ReadDB')
    G.edge('CheckPrice', 'WriteLog')

    G.edge('Checkout', 'Payment')
    G.edge('Payment', 'ReadDB1')
    G.edge('Payment', 'WriteDB')
    G.edge('Payment', 'WriteLog1')

    G.render(
        filename='sample_trace',
        format='pdf',
        cleanup=True
    )

def plot_trace_sample_origin():
    G = Digraph(
        format='pdf',
        graph_attr={'size': '5.0,3.0!', 'nodesep': '1.0'},
        node_attr={'shape': 'rect', 'style': 'rounded,filled', 'width': '0.85', 'height': '0.7'},
        edge_attr={'arrowsize': '0.6', 'penwidth': '1.5'},
        engine='fdp'
    )

    G.node('Checkout', label=get_text_without_times('Checkout', 23), color='lightpink3', pos='0.75,0.5!')
    G.node('CheckPrice0', label=get_text_without_times('CheckPrice', 6), color='lightskyblue4', pos='0.4,0.25!')
    G.node('ReadDB0', label=get_text_without_times('ReadDB', 3), color='aquamarine4', pos='0.4,0!')
    G.node('CheckPrice', label=get_text_without_times('CheckPrice', 5), color='lightskyblue4', pos='0.75,0.25!')
    G.node('ReadDB', label=get_text_without_times('ReadDB', 2), color='aquamarine4', pos='0.6,0!')
    G.node('WriteLog', label=get_text_without_times('WriteLog', 1), color='bisque4', pos='0.8,0.0!')
    G.node('Payment', label=get_text_without_times('Payment', 9), color='navajowhite3', pos='1.2,0.25!')
    G.node('ReadDB1', label=get_text_without_times('ReadDB', 2), color='aquamarine4', pos='1.0,0!')
    G.node('WriteDB', label=get_text_without_times('WriteDB', 4), color='mediumpurple3', pos='1.2,0!')
    G.node('WriteLog1', label=get_text_without_times('WriteLog', 1), color='bisque4', pos='1.4,0!')


    G.edge('Checkout', 'CheckPrice0')
    G.edge('CheckPrice0', 'ReadDB0')

    G.edge('Checkout', 'CheckPrice')
    G.edge('CheckPrice', 'ReadDB')
    G.edge('CheckPrice', 'WriteLog')

    G.edge('Checkout', 'Payment')
    G.edge('Payment', 'ReadDB1')
    G.edge('Payment', 'WriteDB')
    G.edge('Payment', 'WriteLog1')

    G.render(
        filename='sample_trace_origin',
        format='pdf',
        cleanup=True
    )

def plot_trace_sample_structure():
    G = Digraph(
        format='pdf',
        graph_attr={'size': '5.0,3.0!', 'nodesep': '1.0'},
        node_attr={'shape': 'rect', 'style': 'rounded,filled', 'width': '0.85', 'height': '0.7'},
        edge_attr={'arrowsize': '0.6', 'penwidth': '1.5'},
        engine='fdp'
    )

    G.node('Checkout', label=get_text('Checkout', 23), color='lightpink3', pos='1.1,0.6!')

    G.node('CheckPrice', label=get_text_drop_anomaly('CheckPrice', 5.5, 2), color='tomato4', pos='0.75,0.3!', fillcolor='grey90', style='dashed,rounded,filled')
    G.node('ReadDB', label=get_text_drop_anomaly('ReadDB', 2.5, 2), color='tomato4', pos='0.6,0!', fillcolor='grey90', style='dashed,rounded,filled')
    G.node('WriteLog', label=get_text_drop_anomaly('WriteLog', 1), color='tomato4', pos='0.9,0.0!', fillcolor='grey90', style='dashed,rounded,filled')

    G.node('Payment', label=get_text('Payment', 9), color='navajowhite3', pos='1.5,0.3!')
    G.node('ReadDB1', label=get_text('ReadDB', 2), color='aquamarine4', pos='1.2,0!')
    G.node('WriteDB', label=get_text_drop_anomaly('WriteDB', 4), color='tomato4', pos='1.5,0!', fillcolor='grey90', style='dashed,rounded,filled')
    G.node('WriteLog1', label=get_text('WriteLog', 1), color='bisque4', pos='1.8,0!')


    G.edge('Checkout', 'CheckPrice', style='dashed', color='firebrick')
    G.edge('CheckPrice', 'ReadDB', style='dashed')
    G.edge('CheckPrice', 'WriteLog', style='dashed')

    G.edge('Checkout', 'Payment')
    G.edge('Payment', 'ReadDB1')
    G.edge('Payment', 'WriteDB', style='dashed', color='firebrick')
    G.edge('Payment', 'WriteLog1')

    G.render(
        filename='sample_trace_structure',
        format='pdf',
        cleanup=True
    )

def plot_trace_sample_latency():
    G = Digraph(
        format='pdf',
        graph_attr={'size': '5.0,3.0!', 'nodesep': '1.0'},
        node_attr={'shape': 'rect', 'style': 'rounded,filled', 'width': '0.85', 'height': '0.7'},
        edge_attr={'arrowsize': '0.6', 'penwidth': '1.5'},
        engine='fdp'
    )

    G.node('Checkout', label=get_text_latency_anomaly('Checkout', 220), color='lightpink3', pos='1.1,0.6!')
    G.node('CheckPrice', label=get_text_latency_anomaly('CheckPrice', 100, 2), color='lightskyblue4', pos='0.75,0.3!')
    G.node('ReadDB', label=get_text('ReadDB', 2.5, 2), color='aquamarine4', pos='0.6,0!')
    G.node('WriteLog', label=get_text('WriteLog', 1), color='bisque4', pos='0.9,0.0!')
    G.node('Payment', label=get_text_latency_anomaly('Payment', 109), color='navajowhite3', pos='1.5,0.3!')
    G.node('ReadDB1', label=get_text('ReadDB', 2), color='aquamarine4', pos='1.2,0!')
    G.node('WriteDB', label=get_text('WriteDB', 4), color='mediumpurple3', pos='1.5,0!')
    G.node('WriteLog1', label=get_text_latency_anomaly('WriteLog', 100), color='bisque4', pos='1.8,0!')


    G.edge('Checkout', 'CheckPrice')
    G.edge('CheckPrice', 'ReadDB')
    G.edge('CheckPrice', 'WriteLog')

    G.edge('Checkout', 'Payment')
    G.edge('Payment', 'ReadDB1')
    G.edge('Payment', 'WriteDB')
    G.edge('Payment', 'WriteLog1')

    G.render(
        filename='sample_trace_latency',
        format='pdf',
        cleanup=True
    )


if __name__ == '__main__':
    plot_trace_sample_normal()
    plot_trace_sample_origin()
    plot_trace_sample_structure()
    plot_trace_sample_latency()
