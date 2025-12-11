import argparse
import pagerank_calculation as pg

parser = argparse.ArgumentParser(prog='pg', description='iehfhdhf')
subparsers = parser.add_subparsers(dest="command", required=True)

parser_calc_pg = subparsers.add_parser('calc')
parser_calc_pg.add_argument('--p', action='store_true')
parser_calc_pg.add_argument('--l', type=str)
args = parser.parse_args()

def calc_pg(file_likes='Likes.ini', liked=None):
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

    for (name, score) in recommendations:
        print(f"{name}\t{score:.4f}")

if args.command == 'calc':

    if args.p:
        print('hey')
    else:
        calc_pg(liked=args.l)
