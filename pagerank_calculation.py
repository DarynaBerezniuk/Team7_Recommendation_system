"""
Система знайомств на базі персоналізованого PageRank
"""

import random
import numpy as np


def reading_file_people(file_name: str) -> dict[str: list[str]]:
    """
    Функція зчитує файл і виводить словник, де ключ - ім'я людини, а значення - стать і хобі

    >>> _ = open("people_test.txt", "w", encoding="utf-8").write(
    ...     "name_1\\tF\\tМузика\\tПодорожі\\n"
    ...     "name_2\\tM\\tРок\\tХіп-хоп\\n"
    ... )
    >>> data = reading_file_people("people_test.txt")
    >>> data["name_1"]
    ['F', 'Музика', 'Подорожі']
    >>> data["name_2"]
    ['M', 'Рок', 'Хіп-хоп']
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

    >>> _ = open("likes_test.ini", "w", encoding="utf-8").write(
    ...     "(alice, bob)  # Спільний інтерес: Тест\\n"
    ...     "(alice, carol)  # Спільний інтерес: Тест\\n"
    ... )
    >>> graph_creation("likes_test.ini") == {"alice": {"bob", "carol"}}
    True
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

    >>> likes = {"a": {"b", "c"}, "d": {"a"}}
    >>> get_all_users(likes)
    ['a', 'b', 'c', 'd']
    """
    users = set(likes.keys())
    for liked_set in likes.values():
        users.update(liked_set)

    return sorted(list(users))


def create_transition_matrix(likes: dict, users: list) -> np.ndarray:
    """
    Будує матрицю переходів M (транспоновану), де
    M[j, i] = ймовірність перейти з i до j.

    >>> likes = {"a": {"b"}, "b": {"a"}}
    >>> users = get_all_users(likes)
    >>> M = create_transition_matrix(likes, users)
    >>> M.shape
    (2, 2)
    >>> bool({tuple(row) for row in M} == {(0.0, 1.0), (1.0, 0.0)})
    True
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

    Для простого симетричного графа a <-> b важливість вершин майже однакова:

    >>> likes = {"a": {"b"}, "b": {"a"}}
    >>> users = get_all_users(likes)
    >>> M = create_transition_matrix(likes, users)
    >>> pr = calculate_pagerank(M, len(users), max_iterations=50)
    >>> round(float(pr.sum()), 5)
    1.0
    >>> bool(abs(float(pr[0] - pr[1])) < 1e-6)
    True
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

    Якщо немає жодних уподобань, вектор рівномірний:

    >>> users = ["u1", "u2", "u3"]
    >>> k = build_personalization_vector(users, [], [], [])
    >>> [round(float(x), 3) for x in k]
    [0.333, 0.333, 0.333]

    Якщо є вподобання, улюблені користувачі мають більшу вагу:

    >>> k2 = build_personalization_vector(users, ["u1"], ["u3"], [])
    >>> round(float(k2.sum()), 6) == 1.0
    True
    >>> bool(k2[0] > k2[1] > k2[2])
    True
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

    Для однакового графа, але різної персоналізації
    розподіл буде відрізнятися:

    >>> likes = {"a": {"b"}, "b": {"a"}}
    >>> users = get_all_users(likes)
    >>> M = create_transition_matrix(likes, users)
    >>> k1 = build_personalization_vector(users, ["a"], [], [])
    >>> k2 = build_personalization_vector(users, ["b"], [], [])
    >>> pr1 = calculate_personalized_pagerank(M, k1, max_iterations=100)
    >>> pr2 = calculate_personalized_pagerank(M, k2, max_iterations=100)
    >>> bool(pr1[0] > pr1[1] and pr2[1] > pr2[0])
    True
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
    Фільтрує та сортує користувачів за значенням PageRank, виключаючи:
    - поточного користувача
    - уже лайкнутих
    - дизлайкнутих
    - тих, кого вже питали

    >>> users = ["u1", "u2", "u3", "u4"]
    >>> pr = np.array([0.4, 0.3, 0.2, 0.1])
    >>> reccom = generate_recommendations(
    ...     users, pr,
    ...     current_user="u1",
    ...     liked_users=["u2"],
    ...     disliked_users=["u3"],
    ...     asked_users=[]
    ... )
    >>> reccom
    [('u4', 0.1)]
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

    >>> likes = {
    ...     "a": {"b", "c"},
    ...     "b": {"d"},
    ...     "x": {"a"},
    ... }
    >>> build_clan_from_likes(likes, ["a"], [])
    ['b', 'c', 'x']
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
    Розширює список дизлайкнутих користувачів їхніми безпосередніми сусідами
    (тих, кого вони лайкають, і тих, хто лайкає їх).

    >>> likes = {
    ...     "a": {"b"},
    ...     "b": {"c"},
    ...     "d": {"a"},
    ... }
    >>> sorted(extend_disliked_with_neighbors(likes, ["b"]))
    ['a', 'b', 'c']
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
    Виводить профіль людини (стать і хобі) у форматованому рядку.

    >>> _ = open("hobbies_test.ini", "w", encoding="utf-8").write(
    ...     "name_1\\tF\\tМузика\\tПодорожі\\n"
    ... )
    >>> suggest_people("name_1", "hobbies_test.ini")
    'Профіль: name_1, стать : F, хобі: Музика, Подорожі'
    """
    dict_people = reading_file_people(file_name)
    people = dict_people[user_name]
    sex = people[0]
    hobby = people[1:]

    return f'Профіль: {user_name}, стать : {sex}, хобі: {', '.join(hobby)}'

def write_pagerank_in_file(recommendations: list[tuple[str, float]]):
    """
    Динамічно записує значення PageRank для кожного кандидата у файл result.txt

    Аргументи:
        recommendations list[tuple[str, float]]: список з кортежів,
                                                 які репрезентують ім'я та
                                                 pagerank значення

    Зауваження: ця функція нічого не повертає, а записує у файл result.txt
                отримані значення
    """
    with open('output/result.txt', 'w', encoding='utf-8') as f:
        for (name, score) in recommendations:
            f.write(f"{name}\t{score:.4f}\n")

def main(file_likes: str):
    """
    Основна функція, яка динамічно розраховує значення PageRank для користувача
    на основі лайкнутих та дизлайкнутих людей
    """
    likes = graph_creation(file_likes)
    users = get_all_users(likes)
    matrix = create_transition_matrix(likes, users)

    if not users:
        print("У графі немає жодного користувача.")
        return

    liked_users = []
    disliked_users = []
    asked_users = []

    current_candidate = random.choice(users)

    while True:
        if len(asked_users) == len(users):
            print("\nМи показали всі доступні профілі: )")
            break

        if current_candidate in asked_users:
            remaining = [u for u in users if u not in asked_users]
            if not remaining:
                print("\nМи показали всі доступні профілі: )")
                break
            current_candidate = random.choice(remaining)
            continue

        print(suggest_people(current_candidate, 'data/hobbies.ini'))
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

        recs_without_del = generate_recommendations(
            users,
            pageranks,
            current_user=None,
            liked_users=None,
            disliked_users=None,
            asked_users=None,
        )
        write_pagerank_in_file(recs_without_del)

        print("==========================================")
        for idx, (name, score) in enumerate(recommendations[:10], start=1):
            print(f"{idx:2d}. {name}  |  PR = {score:.4f}")

        current_candidate = recommendations[0][0]
    print("====================================")
    print("Твої лайки:", liked_users if liked_users else "немає")
    print("Твої дизлайки:", disliked_users if disliked_users else "немає")


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    main(r"data\likes.ini")
