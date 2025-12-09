import argparse
import added_hobby_output as ahb

parser = argparse.ArgumentParser(description='')

subparser = parser.add_subparsers(dest='command', help='')

parser_graph_creation = subparser.add_parser('create_graph', help='')
parser_graph_creation.add_argument('f', '--filename', type=str, required=True, help='')
parser_graph_creation.add_argument('--getusers', '-g', type=str, help='')