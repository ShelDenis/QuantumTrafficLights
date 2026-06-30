import pennylane as qml
from pennylane import numpy as np
import networkx as nx


def create_traffic_graph_with_routes(filename):
    G = nx.Graph()

    edges = []
    all_nodes = set()

    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            u, v, weight = int(parts[0]), int(parts[1]), float(parts[2])
            edges.append((u, v, weight))
            all_nodes.add(u)
            all_nodes.add(v)

    G.add_nodes_from(sorted(all_nodes))

    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    print("Граф загружен:")
    print(f"   Вершин: {len(G.nodes())}")
    print(f"   Рёбер: {len(G.edges())}")
    print(f"   Вершины: {sorted(G.nodes())}")

    return G


def define_routes(G):
    routes = [
        {
            'source': 1,  # старт
            'destination': 4,  # финиш
            'path': [1, 2, 3, 4],  # Оптимальный путь
            'priority': 1.0,  # Приоритет маршрута
            'traffic_volume': 100  # Интенсивность движения (машин/час)
        },
        {
            'source': 1,
            'destination': 9,
            'path': [1, 5, 6, 7, 8, 9],
            'priority': 0.5,
            'traffic_volume': 50
        }
    ]

    return routes


def build_node_to_qubit_map(graph):
    all_nodes = sorted(list(graph.nodes()))
    node_to_qubit = {node: idx for idx, node in enumerate(all_nodes)}
    qubit_to_node = {idx: node for node, idx in node_to_qubit.items()}

    print(f"Всего вершин (и соот-но кубитов): {len(all_nodes)}")

    return node_to_qubit, qubit_to_node, all_nodes


def build_cost_hamiltonian_with_routes(G, routes, node_to_qubit):
    coeffs = []
    obs = []

    for route in routes:
        path = route['path']
        priority = route['priority']
        volume = route['traffic_volume']

        for i in range(len(path) - 1):
            light_i = path[i]
            light_j = path[i + 1]

            if not G.has_edge(light_i, light_j):
                continue

            if light_i not in node_to_qubit or light_j not in node_to_qubit:
                continue

            qi = node_to_qubit[light_i]
            qj = node_to_qubit[light_j]

            travel_time = G[light_i][light_j]['weight']
            weight = (priority * volume * travel_time) / 1000.0

            coeffs.append(weight)
            obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))

    for u, v in G.edges():
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]
        coeffs.append(0.05)
        obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))

    H = qml.Hamiltonian(coeffs, obs)
    return H


def evaluate_route(G, route, phase_offsets, cycle_time=2):
    path = route['path']
    total_time = 0
    current_time = 0

    print(f"\nМаршрут: {' → '.join(map(str, path))}")

    for i in range(len(path) - 1):
        light_current = path[i]
        light_next = path[i + 1]

        travel_time = G[light_current][light_next]['weight']

        # Фазовое смещение следующего светофора (в минутах)
        offset_next = phase_offsets[light_next] * (cycle_time / 2)

        # Когда машина прибудет на следующий светофор
        arrival_time = current_time + travel_time

        # Определяем фазу светофора в момент прибытия
        # Фаза меняется каждые cycle_time/2 секунд
        phase_at_arrival = int((arrival_time + offset_next) / (cycle_time / 2)) % 2

        # Фаза 0 - зеленый, 1 - красный
        if phase_at_arrival == 1:
            # Нужно ждать до следующего зеленого
            wait_time = cycle_time / 2 - ((arrival_time + offset_next) % (cycle_time / 2))
            print(f"  Перекресток {light_next}: КРАСНЫЙ, ожидание {wait_time:.1f} мин")
            total_time += wait_time
            current_time = arrival_time + wait_time
        else:
            print(f"  Перекресток {light_next}: ЗЕЛЕНЫЙ, проезд {travel_time} мин")
            current_time = arrival_time

        total_time += travel_time

    print(f"  Общее время в пути: {total_time:.1f} мин")
    return total_time


def optimize_traffic_with_routes(graph, routes, depth, initial_gamma, initial_beta):
    # Строим маппинг (забиваем кубиты)
    node_to_qubit, qubit_to_node, all_nodes = build_node_to_qubit_map(graph)
    num_qubits = len(all_nodes)

    print(f"\nТранспортная сеть: {num_qubits} светофоров")
    print(f"Количество маршрутов: {len(routes)}")
    print(f"Глубина QAOA: {depth}\n")

    dev = qml.device("default.qubit", wires=num_qubits, shots=1000)

    H = build_cost_hamiltonian_with_routes(graph, routes, node_to_qubit)

    @qml.qnode(dev)
    def qaoa_circuit(gamma, beta):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)

        for layer in range(depth):
            qml.ApproxTimeEvolution(H, gamma[layer], 1)
            for i in range(num_qubits):
                qml.RX(2 * beta[layer], wires=i)

        return qml.expval(H)

    def cost_function(params):
        gamma = params[:depth]
        beta = params[depth:]
        return qaoa_circuit(gamma, beta)

    init_params = np.concatenate([initial_gamma, initial_beta])
    init_params = np.array(init_params, requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    params = init_params

    print("Запуск оптимизации...")
    for i in range(50):
        params = opt.step(cost_function, params)
        if (i + 1) % 10 == 0:
            cost = cost_function(params)
            print(f"Шаг {i + 1:3d} | Стоимость: {cost:.4f}")

    optimal_gamma = params[:depth]
    optimal_beta = params[depth:]
    final_cost = cost_function(params)

    # Получаем фазовые смещения
    @qml.qnode(dev)
    def get_final_state():
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for layer in range(depth):
            qml.ApproxTimeEvolution(H, optimal_gamma[layer], 1)
            for i in range(num_qubits):
                qml.RX(2 * optimal_beta[layer], wires=i)
        return qml.probs(wires=range(num_qubits))

    probs = get_final_state()
    best_state = np.argmax(probs)

    phase_offsets = {}
    for qubit_idx in range(num_qubits):
        bit = (best_state >> qubit_idx) & 1
        node = qubit_to_node[qubit_idx]  # обратный маппинг
        phase_offsets[node] = bit

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)
    print(f"Значение целевой функции: {final_cost:.4f}")
    print(f"Оптимальные γ: {optimal_gamma}")
    print(f"Оптимальные β: {optimal_beta}")

    print("\nФАЗОВЫЕ СМЕЩЕНИЯ:")
    for node in sorted(phase_offsets.keys()):
        phase = phase_offsets[node]
        phase_desc = "Фаза 0 (базовая)" if phase == 0 else "Фаза 1 (сдвиг)"
        print(f"  Вершина {node:3d}: {phase_desc}")

    return phase_offsets, final_cost, optimal_gamma, optimal_beta


if __name__ == "__main__":
    G = create_traffic_graph_with_routes('examples/very_simple.txt')

    routes = define_routes(G)

    num_phases = 2
    depth = 2
    initial_gamma = [0.5, 0.3]
    initial_beta = [0.2, 0.4]

    phase_offsets, cost, opt_gamma, opt_beta = optimize_traffic_with_routes(
        graph=G,
        routes=routes,
        depth=depth,
        initial_gamma=initial_gamma,
        initial_beta=initial_beta
    )

    print("\n" + "=" * 60)
    print("ОЦЕНКА МАРШРУТОВ")
    print("=" * 60)

    for route in routes:
        time = evaluate_route(G, route, phase_offsets)
        print(f"Общее время маршрута: {time:.1f} мин")