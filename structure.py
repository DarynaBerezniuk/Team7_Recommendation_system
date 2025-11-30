import numpy as np
def reading_file_hobby(file_name: str) -> dict:
    result_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            elements = line.split(',')
            result_dict.setdefault(elements[0], [])
            result_dict[elements[0]].append(tuple(elements[1:]))

    return result_dict

def reading_file_people(file_name: str) -> dict:
    result_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            elements = line.split('\t')
            result_dict.setdefault(elements[0], [])
            result_dict[elements[0]].append(elements[1:])

    return result_dict

def graph_creation(file_name: str) -> dict:
    result_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('# Спільний інтерес')
            line = line[0][1:-3]
            line = line.split(',')
            result_dict.setdefault(line[0], set())
            result_dict[line[0]].add(line[1])
    return result_dict\

def get_all_users(likes: dict[str, set[str]]) -> list[str]:
    """
    Створює відсортований список усіх унікальних користувачів з графа.
    """
    users = set(likes.keys())
    for liked_set in likes.values():
        users.update(liked_set)

    return sorted(list(users))

def create_transition_matrix(likes: dict, users: list) -> list:
    num_users = len(users)
    matrix = np.zeros((num_users, num_users))

    for i, name_1 in enumerate(users):
        for j, name_2 in enumerate(users):
            if name_1 in likes:
                if name_2 in likes[name_1]:
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
    tolerance: float = 1e-6 ) -> np.ndarray:
    """
    Обчислює вектору PageRank ітеративним методом.

    matrix: Транспонована матриця переходу.
    alpha: Коефіцієнт демпфування (ймовірність переходу по лінку).
    """

    vector = np.ones(num_users) / num_users

    #(1 - alpha) * (1/N)
    vector_teleport = np.ones(num_users) * (1 - alpha) / num_users

    for _ in range(max_iterations):
        previous_vector = vector.copy()

        # matrix @ previous_vector - це матричне множення
        rank_mass = alpha * (matrix @ previous_vector)
        vector = rank_mass + vector_teleport

        # ||PR - PR_prev|| має бути менша за tolerance
        if np.linalg.norm(vector - previous_vector, ord=1) < tolerance:
            break

    return vector

def generate_recommendations(likes: list, users: list) -> list[tuple[str, float]]:
    """
    Основна функція, що об'єднує всі кроки для отримання рейтингів.
    """
    num_users = len(users)
    matrix = create_transition_matrix(likes, users)

    pageranks = calculate_pagerank(matrix, num_users)

    results = list(zip(users, pageranks.tolist()))

    results.sort(key=lambda item: item[1], reverse=True)

    return results

def main(file_likes: str, finaly_file:str):
    likes = graph_creation(file_likes)
    users = get_all_users(likes)
    result = generate_recommendations(likes, users)

    with open(finaly_file, 'w', encoding='utf-8') as file:
        for element in result:
            file.write(element[0].strip())
            file.write(' ')
            file.write(str(round(element[1], 4)))
            file.write('\n')

main('Likes.ini', 'result')
