"""
Команди для інтеракції з модулем pagerank_calculation через командний рядок
"""

import os
import argparse
import pagerank_calculation as pg

parser = argparse.ArgumentParser(prog='pg', description='Executes commands such as: \
                                 ')
subparsers = parser.add_subparsers(dest="command", required=True)

parser_calc_pg = subparsers.add_parser('calc', help='calculates PageRank with multiple \
                                       conditions. Default: calculates PageRank without \
                                       caring for liked or disliked people')
parser_calc_pg.add_argument('-l', '--liked', metavar='', type=str, help='use it if you want to \
                            calculate PageRank accordingly to the person you liked')
parser_calc_pg.add_argument('-w', '--write', action='store_true', help='use it if you want \
                            to write the calculation results in a file called result.txt')

parser_calc_pg = subparsers.add_parser('run', help='runs website to visualise algorithm')
args = parser.parse_args()

def calc_pg(file_likes=r'data\likes.ini', liked=None, write=False):
    """
    Розраховує значення PageRank для кожної людини
    
        file_likes str: шлях до файлу з візуалізацією ребер
        liked str: ім'я людини, відносно якої треба порахувати PageRank
        write bool: True, якщо треба записати функцію у файл

    Зауваження: ця функція нічого не повертає, але записує у файл список людей
                за рейтингом PageRank, або виводить цей список у термінал
    """
    likes = pg.graph_creation(file_likes)
    users = pg.get_all_users(likes)
    matrix = pg.create_transition_matrix(likes, users)

    liked_users = [liked]
    disliked_users = []
    asked_users = [liked]

    clan_users = pg.build_clan_from_likes(likes, liked_users, disliked_users)

    extended_disliked = pg.extend_disliked_with_neighbors(likes, disliked_users)

    personalization = pg.build_personalization_vector(
        users=users,
        liked_users=liked_users,
        disliked_users=list(extended_disliked),
        clan_users=clan_users,
    )

    pageranks = pg.calculate_personalized_pagerank(matrix, personalization)

    recommendations = pg.generate_recommendations(
        users,
        pageranks,
        current_user=None,
        liked_users=liked_users,
        disliked_users=disliked_users,
        asked_users=asked_users,
    )

    if write is True:
        pg.write_pagerank_in_file(recommendations)
    else:
        for (name, score) in recommendations:
            print(f"{name}\t{score:.4f}")

if args.command == 'calc':
    if args.l and args.w:
        calc_pg(liked=args.liked, write=args.write)
    elif args.l:
        calc_pg(liked=args.liked)
    else:
        calc_pg()
elif args.command == 'run':
    os.system('streamlit run app.py')
