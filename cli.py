import argparse
import pagerank_calculation as ahb

parser = argparse.ArgumentParser(description='iehfhdhf')

subparser = parser.add_subparsers(dest='command', help='jkdfhds')

parser_graph_creation = subparser.add_parser('create_graph', help='')
parser_graph_creation.add_argument('f', '--filename', type=str, required=True, help='')
parser_graph_creation.add_argument('--getusers', '-g', type=str, help='')

args = parser.parse_args()