"""
Система знайомств на базі персоналізованого PageRank
"""

import random
import numpy as np


# def reading_file_hobby(file_name: str) -> dict:
#     result_dict = {}
#     with open(file_name, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             elements = line.split(',')
#             result_dict.setdefault(elements[0], [])
#             result_dict[elements[0]].append(tuple(elements[1:]))

#     return result_dict


def reading_file_people(file_name: str) -> dict:
    """
    Функція зчитує файл і виводить словник, де ключ - ім'я людини, а значення- стать і хобі
    """
    result_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            elements = line.split('\t')
            result_dict.setdefault(elements[0], [])
            result_dict[elements[0]].extend(elements[1:])

    return result_dict


def graph_creation(file_name: str) -> dict:
    """
    Читає Likes.ini формату:
    (name_1, name_3)  # Спільний інтерес: Подорожі

    Повертає словник:
    {
        "name_1": {"name_3", ...},
        ...
    }
    """
    result_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('# Спільний інтерес')[0].strip()
            if not line:
                continue
            inner = line[1:-1]
            parts = [p.strip() for p in inner.split(',')]
            if len(parts) != 2:
                continue
            a, b = parts
            result_dict.setdefault(a, set())
            result_dict[a].add(b)

    return result_dict


def get_all_users(likes: dict[str, set[str]]) -> list[str]:
    """
    Створює відсортований список усіх унікальних користувачів з графа.
    """
    users = set(likes.keys())
    for liked_set in likes.values():
        users.update(liked_set)

    return sorted(list(users))


def create_transition_matrix(likes: dict, users: list) -> np.ndarray:
    """
    Будує матрицю переходів M (транспоновану), де
    M[j, i] = ймовірність перейти з i до j.
    """
    num_users = len(users)
    matrix = np.zeros((num_users, num_users))

    for i, name_1 in enumerate(users):
        for j, name_2 in enumerate(users):
            if name_1 in likes and name_2 in likes[name_1]:
                matrix[i, j] = 1

    out_degree = matrix.sum(axis=1)

    result_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        if out_degree[i] > 0:
            result_matrix[i, :] = matrix[i, :] / out_degree[i]

    return result_matrix.T


def calculate_pagerank(
    matrix: np.ndarray,
    num_users: int,
    alpha: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    Функція реалізує ітеративний метод для знаходження вектора PageRank,
    який слугує мірою відносної важливості кожного вузла в графі,
    представленому вхідною матрицею.

    """
    vector = np.ones(num_users) / num_users
    vector_teleport = np.ones(num_users) * (1 - alpha) / num_users

    for _ in range(max_iterations):
        previous_vector = vector.copy()

        rank_mass = alpha * (matrix @ previous_vector)
        vector = rank_mass + vector_teleport

        if np.linalg.norm(vector - previous_vector, ord=1) < tolerance:
            break

    return vector


def build_personalization_vector(
    users: list[str],
    liked_users: list[str],
    disliked_users: list[str],
    clan_users: list[str],
) -> np.ndarray:
    """
    Ця функція призначена для створення персоналізованого вектора розподілу ймовірностей
    (або вектора телепортації) на основі вподобань і соціальних зв'язків поточного
    користувача в системі рекомендацій чи алгоритмі PageRank з персоналізацією.
    """
    num_users = len(users)
    scores = np.zeros(num_users)

    liked_set = set(liked_users)
    disliked_set = set(disliked_users)
    clan_set = set(clan_users)

    LIKE_WEIGHT = 1.0
    DISLIKE_WEIGHT = -0.7
    CLAN_WEIGHT = 0.3

    for i, name in enumerate(users):
        score = 0.0

        if name in clan_set:
            score += CLAN_WEIGHT

        if name in liked_set:
            score += LIKE_WEIGHT

        if name in disliked_set:
            score += DISLIKE_WEIGHT

        scores[i] = score

    if np.all(scores == 0):
        return np.ones(num_users) / num_users

    min_score = scores.min()
    if min_score < 0:
        scores = scores - min_score

    total = scores.sum()
    if total == 0:
        return np.ones(num_users) / num_users

    k = scores / total
    return k


def calculate_personalized_pagerank(
    matrix: np.ndarray,
    personalization: np.ndarray,
    alpha: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Функція обчислює Персоналізований PageRank
    PR_{t+1} = α * M @ PR_t  +  (1 - α) * k
    """
    num_users = len(personalization)

    vector = np.ones(num_users) / num_users
    vector_teleport = (1 - alpha) * personalization

    for _ in range(max_iterations):
        previous_vector = vector.copy()

        rank_mass = alpha * (matrix @ previous_vector)
        vector = rank_mass + vector_teleport

        if np.linalg.norm(vector - previous_vector, ord=1) < tolerance:
            break

    return vector


def generate_recommendations(
    users: list[str],
    pageranks: np.ndarray,
    current_user: str | None = None,
    liked_users: list[str] | None = None,
    disliked_users: list[str] | None = None,
    asked_users: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    1
    """
    if liked_users is None:
        liked_users = []
    if disliked_users is None:
        disliked_users = []
    if asked_users is None:
        asked_users = []

    results = list(zip(users, pageranks.tolist()))

    liked_set = set(liked_users)
    disliked_set = set(disliked_users)
    asked_set = set(asked_users)

    filtered = []
    for name, score in results:
        if score <= 0:
            continue
        if current_user is not None and name == current_user:
            continue
        if name in liked_set:
            continue
        if name in disliked_set:
            continue
        if name in asked_set:
            continue
        filtered.append((name, score))

    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered


def build_clan_from_likes(
    likes,
    liked_users,
    disliked_users,
):
    """
    Кінцевий результат є списком рекомендованих користувачів,
    які знаходяться на відстані двох кроків від поточного користувача в графі вподобань.
    """
    clan_set: set[str] = set()

    liked_set = set(liked_users)
    disliked_set = set(disliked_users)

    for u in liked_users:
        clan_set.update(likes.get(u, set()))

    for v, neighs in likes.items():
        for u in liked_users:
            if u in neighs:
                clan_set.add(v)

    clan_set.difference_update(liked_set)
    clan_set.difference_update(disliked_set)

    return sorted(clan_set)


def extend_disliked_with_neighbors(likes, disliked_users):
    """
    1
    """
    extended: set[str] = set(disliked_users)

    for d in disliked_users:
        extended.update(likes.get(d, set()))

    for u, neighs in likes.items():
        for d in disliked_users:
            if d in neighs:
                extended.add(u)

    return extended

def suggest_people(user_name, file_name:str) -> dict:
    """
    Виводить стать та вподобання людини
    """
    dict_people = reading_file_people(file_name)
    people = dict_people[user_name]
    sex = people[0]
    hobby = people[1:]

    return f'Профіль: {user_name}, стать : {sex}, хобі: {' ,'.join(hobby)}'


def main(file_likes: str):
    """
    main func
    """
    likes = graph_creation(file_likes)
    users = get_all_users(likes)
    matrix = create_transition_matrix(likes, users)

    if not users:
        print("У графі немає жодного користувача")
        return


    liked_users = []
    disliked_users = []
    asked_users = []

    current_candidate = random.choice(users)

    while True:
        if len(asked_users) == len(users):
            print("\nМи показали всі доступні профілі :)")
            break

        if current_candidate in asked_users:
            remaining = [u for u in users if u not in asked_users]
            if not remaining:
                print("\nМи показали всі доступні профілі :)")
                break
            current_candidate = random.choice(remaining)
            continue

        print(suggest_people(current_candidate, 'File_1.ini'))
        ans = input("Подобається? [y/n/Enter щоб завершити]: ").strip().lower()

        if ans == "":
            break
        elif ans == "y":
            if current_candidate not in liked_users:
                liked_users.append(current_candidate)
        elif ans == "n":
            if current_candidate not in disliked_users:
                disliked_users.append(current_candidate)
        else:
            continue

        asked_users.append(current_candidate)

        if not liked_users and not disliked_users:
            remaining = [u for u in users if u not in asked_users]
            if not remaining:
                break
            current_candidate = random.choice(remaining)
            continue

        clan_users = build_clan_from_likes(likes, liked_users, disliked_users)

        extended_disliked = extend_disliked_with_neighbors(likes, disliked_users)

        personalization = build_personalization_vector(
            users=users,
            # current_user="",
            liked_users=liked_users,
            disliked_users=list(extended_disliked),
            clan_users=clan_users,
        )

        pageranks = calculate_personalized_pagerank(matrix, personalization)

        recommendations = generate_recommendations(
            users,
            pageranks,
            current_user=None,
            liked_users=liked_users,
            disliked_users=disliked_users,
            asked_users=asked_users,
        )

        if not recommendations:
            break

        print("==========================================")
        for idx, (name, score) in enumerate(recommendations[:10], start=1):
            print(f"{idx:2d}. {name}  |  PR = {score:.4f}")

        current_candidate = recommendations[0][0]
    print("====================================")
    print("Твої лайки:", liked_users if liked_users else "немає")
    print("Твої дизлайки:", disliked_users if disliked_users else "немає")


if __name__ == "__main__":
    main("Likes.ini")
