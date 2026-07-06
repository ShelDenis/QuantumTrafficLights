import json
import random
import math
from collections import defaultdict, deque


def load_graph_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data['nodes']
    edges = data['edges']

    node_ids = sorted([int(nid) for nid in nodes.keys()])

    graph = defaultdict(dict)
    edge_weights = {}

    for edge in edges:
        u = edge['start']
        v = edge['end']
        travel_time = edge['current_travel_time']
        fitness = edge['fitness']
        weight = travel_time / (fitness / 100 + 1)
        graph[u][v] = weight
        graph[v][u] = weight
        edge_weights[(u, v)] = fitness
        edge_weights[(v, u)] = fitness

    return graph, node_ids, edges, edge_weights


def find_diverse_paths(graph, node_ids, edge_weights, max_paths=50):
    vertex_importance = {}
    for node in node_ids:
        degree = len(graph[node])
        total_fitness = sum(edge_weights.get((node, neighbor), 50)
                            for neighbor in graph[node])
        vertex_importance[node] = degree * total_fitness

    important_vertices = sorted(vertex_importance.items(), key=lambda x: x[1], reverse=True)

    groups = {}
    for node, imp in important_vertices:
        group_id = node // 10
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(node)

    all_paths = []

    group_ids = list(groups.keys())

    for i, group1 in enumerate(group_ids):
        for group2 in group_ids[i + 1:]:
            start_candidates = groups[group1][:3]
            end_candidates = groups[group2][:3]

            for start in start_candidates:
                for end in end_candidates:
                    paths = find_paths_between(graph, start, end, min_length=3, max_length=6)

                    for path in paths:
                        if len(set(path)) >= 4:
                            all_paths.append(path)

    top_vertices = [v for v, _ in important_vertices[:10]]

    for i, start in enumerate(top_vertices):
        for end in top_vertices[i + 1:]:
            paths = find_paths_between(graph, start, end, min_length=3, max_length=6)
            for path in paths:
                if len(set(path)) >= 4:
                    all_paths.append(path)

    unique_paths = []
    for path in all_paths:
        path_set = set(path)
        is_duplicate = False

        for existing_path in unique_paths:
            existing_set = set(existing_path)
            similarity = len(path_set & existing_set) / len(path_set | existing_set)
            if similarity > 0.8:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_paths.append(path)

    path_scores = []
    for path in unique_paths:
        score = evaluate_path_diversity(path, graph, edge_weights, vertex_importance)
        path_scores.append((path, score))

    path_scores.sort(key=lambda x: x[1], reverse=True)

    return [path for path, _ in path_scores[:max_paths]]


def find_paths_between(graph, start, end, min_length=3, max_length=6):
    all_paths = []
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()

        if current == end and len(path) >= min_length:
            all_paths.append(path)
            if len(all_paths) >= 5:
                break

        if len(path) < max_length:
            for neighbor in graph[current]:
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

    return all_paths


def evaluate_path_diversity(path, graph, edge_weights, vertex_importance):
    score = 0

    if 4 <= len(path) <= 5:
        score += 10
    elif len(path) == 3:
        score += 5
    elif len(path) > 5:
        score += 7

    importance_variance = 0
    if len(path) >= 2:
        importances = [vertex_importance.get(v, 0) for v in path]
        importance_variance = max(importances) - min(importances)
        score += importance_variance * 0.01

    edge_qualities = []
    for i in range(len(path) - 1):
        fitness = edge_weights.get((path[i], path[i + 1]), 50)
        edge_qualities.append(fitness)

    if edge_qualities:
        avg_quality = sum(edge_qualities) / len(edge_qualities)
        score += avg_quality * 0.1

    degrees = [len(graph[v]) for v in path]
    degree_variance = max(degrees) - min(degrees) if degrees else 0
    score += degree_variance * 2

    return score


def select_optimal_routes_evolutionary(candidate_paths, n_routes=5, max_vertices=12):
    if len(candidate_paths) < n_routes:
        return candidate_paths

    population_size = 200
    generations = 300

    population = []

    for _ in range(population_size // 2):
        used_starts = set()
        selected_indices = []

        available = list(range(len(candidate_paths)))
        random.shuffle(available)

        for idx in available:
            if len(selected_indices) >= n_routes:
                break
            path = candidate_paths[idx]
            if path[0] not in used_starts or len(used_starts) >= 3:
                selected_indices.append(idx)
                used_starts.add(path[0])

        if len(selected_indices) == n_routes:
            population.append(selected_indices)

    for _ in range(population_size - len(population)):
        indices = random.sample(range(len(candidate_paths)), n_routes)
        population.append(indices)

    best_solution = None
    best_fitness = -float('inf')

    for generation in range(generations):
        fitness_scores = []

        for indices in population:
            paths = [candidate_paths[i] for i in indices]

            unique_vertices = set()
            for path in paths:
                unique_vertices.update(path)

            vertex_count = len(unique_vertices)

            if vertex_count > max_vertices:
                vertex_penalty = (vertex_count - max_vertices) ** 2 * 5000
            else:
                vertex_penalty = 0

            if 8 <= vertex_count <= 12:
                vertex_bonus = 20
            elif 6 <= vertex_count < 8:
                vertex_bonus = 10
            else:
                vertex_bonus = 0

            intersection_score = 0
            intersection_pairs = 0

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    common = set(paths[i]) & set(paths[j])
                    if common:
                        intersection_pairs += 1
                        similarity = len(common) / min(len(paths[i]), len(paths[j]))
                        if 0.2 <= similarity <= 0.8:
                            intersection_score += 15
                        elif similarity < 0.2:
                            intersection_score += 5
                        else:
                            intersection_score += 2

            starts = set(path[0] for path in paths)
            ends = set(path[-1] for path in paths)
            diversity_score = (len(starts) + len(ends)) * 8

            path_lengths = [len(path) for path in paths]
            length_diversity = max(path_lengths) - min(path_lengths) if path_lengths else 0

            similarity_penalty = 0
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    sim = len(set(paths[i]) & set(paths[j])) / len(set(paths[i]) | set(paths[j]))
                    if sim > 0.7:
                        similarity_penalty += (sim - 0.7) * 30

            fitness = (intersection_score + diversity_score + vertex_bonus +
                       length_diversity * 3 - vertex_penalty - similarity_penalty)

            fitness_scores.append(fitness)

        best_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx]

        new_population = []

        elite_count = max(2, population_size // 10)
        elite_indices = sorted(range(len(fitness_scores)),
                               key=lambda i: fitness_scores[i],
                               reverse=True)[:elite_count]
        new_population.extend([population[i] for i in elite_indices])

        while len(new_population) < population_size:
            tournament_size = 3
            parent1 = tournament_select(population, fitness_scores, tournament_size)
            parent2 = tournament_select(population, fitness_scores, tournament_size)

            child = []
            used_indices = set()

            crossover_point = n_routes // 2
            for i in range(crossover_point):
                child.append(parent1[i])
                used_indices.add(parent1[i])

            for idx in parent2:
                if len(child) >= n_routes:
                    break
                if idx not in used_indices:
                    child.append(idx)
                    used_indices.add(idx)

            while len(child) < n_routes:
                available = [i for i in range(len(candidate_paths)) if i not in used_indices]
                if available:
                    new_idx = random.choice(available)
                    child.append(new_idx)
                    used_indices.add(new_idx)
                else:
                    break

            if random.random() < 0.15:
                mutate_idx = random.randint(0, len(child) - 1)
                available = [i for i in range(len(candidate_paths))
                             if i not in child or i == child[mutate_idx]]
                if available:
                    child[mutate_idx] = random.choice(available)

            new_population.append(child)

        population = new_population

    if best_solution:
        return [candidate_paths[i] for i in best_solution]
    else:
        return candidate_paths[:n_routes]


def tournament_select(population, fitness_scores, k=3):
    tournament_indices = random.sample(range(len(population)), k)
    return population[max(tournament_indices, key=lambda i: fitness_scores[i])]


def generate_route_code(routes):
    code = "def define_routes(G):\n"
    code += "    routes = [\n"

    for route in routes:
        code += "        {\n"
        code += f"            'source': {route['source']},\n"
        code += f"            'destination': {route['destination']},\n"
        code += f"            'path': {route['path']},\n"
        code += f"            'priority': {route['priority']},\n"
        code += f"            'traffic_volume': {route['traffic_volume']}\n"
        code += "        },\n"

    code += "    ]\n"
    code += "    return routes\n"

    return code


if __name__ == "__main__":
    graph, node_ids, edges, edge_weights = load_graph_from_json('traffic_graph_final.json')

    print(f"{len(node_ids)} вершин, {len(edges)} рёбер")

    diverse_paths = find_diverse_paths(graph, node_ids, edge_weights, max_paths=60)
    print(f"Найдено {len(diverse_paths)} разнообразных путей")

    selected_paths = select_optimal_routes_evolutionary(diverse_paths, n_routes=5, max_vertices=12)

    edge_id_map = {}
    for edge in edges:
        u, v = edge['start'], edge['end']
        edge_id_map[(u, v)] = edge.get('edge_id', f'edge_{u}_{v}')
        edge_id_map[(v, u)] = edge.get('edge_id', f'edge_{u}_{v}')

    routes = []
    for path in selected_paths:
        total_fitness = sum(edge_weights.get((path[i], path[i + 1]), 50)
                            for i in range(len(path) - 1))
        avg_fitness = total_fitness / max(1, len(path) - 1)

        priority = 0.5 + (avg_fitness / 100) * 1.5
        traffic_volume = int(50 + avg_fitness * 1.5)

        routes.append({
            'source': path[0],
            'destination': path[-1],
            'path': path,
            'priority': round(min(priority, 2.0), 1),
            'traffic_volume': min(traffic_volume, 200)
        })

    print()
    print("Оптимальные маршруты")
    print()

    unique_vertices = set()
    for i, route in enumerate(routes):
        print(f"\nМаршрут {i + 1}:")
        print(f"  Путь: {' -> '.join(map(str, route['path']))}")
        print(f"  Длина: {len(route['path'])} вершин")
        print(f"  Приоритет: {route['priority']}")
        print(f"  Интенсивность: {route['traffic_volume']}")

        print(f"  Рёбра:")
        for j in range(len(route['path']) - 1):
            u = route['path'][j]
            v = route['path'][j + 1]
            eid = edge_id_map.get((u, v), f'?')
            fitness = edge_weights.get((u, v), '?')
            print(f"    edge {eid}: ({u}, {v}) [fitness={fitness}]")

        unique_vertices.update(route['path'])

    print(f"\nУникальные вершины: {len(unique_vertices)}")
    print(f"Вершины: {sorted(unique_vertices)}")

    print("\nПересечения:")
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            common = set(routes[i]['path']) & set(routes[j]['path'])
            if common:
                print(f"  Маршруты {i + 1} и {j + 1}: {len(common)} общих вершин - {sorted(common)}")
    route_code = generate_route_code(routes)
    print(route_code)

    with open('routes_for_quantum.py', 'w', encoding='utf-8') as f:
        f.write(route_code)