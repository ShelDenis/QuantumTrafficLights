import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import matplotlib.pyplot as plt  
from routes_for_quantum import define_routes
import json

from geopy.distance import geodesic


def create_traffic_graph_with_routes(filename):
    G = nx.Graph()

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    routes = define_routes(None)

    used_nodes = set()
    for route in routes:
        used_nodes.update(route['path'])

    for node_id in used_nodes:
        node_info = data['nodes'][str(node_id)]
        G.add_node(int(node_id), lat=node_info['lat'], lon=node_info['lon'])

    for edge in data['edges']:
        u = edge['start']
        v = edge['end']

        if u in used_nodes and v in used_nodes:
            node_u = data['nodes'][str(u)]
            node_v = data['nodes'][str(v)]

            dist_km = geodesic(
                (node_u['lat'], node_u['lon']),
                (node_v['lat'], node_v['lon'])
            ).km

            current_speed = edge.get('current_speed', None)

            if current_speed and current_speed > 0:
                travel_time = (dist_km / current_speed) * 60
            else:
                travel_time = (dist_km / 35) * 60

            travel_time = max(0.1, travel_time)

            G.add_edge(u, v, weight=travel_time)

    print("Граф загружен:")
    print(f"   Вершин: {len(G.nodes())}")
    print(f"   Рёбер: {len(G.edges())}")
    print(f"   Вершины: {sorted(G.nodes())}")

    return G


def build_node_to_qubit_map(graph):
    all_nodes = sorted(list(graph.nodes()))
    node_to_qubit = {node: idx for idx, node in enumerate(all_nodes)}
    qubit_to_node = {idx: node for node, idx in node_to_qubit.items()}

    print(f"Всего вершин (и соот-но кубитов): {len(all_nodes)}")

    return node_to_qubit, qubit_to_node, all_nodes


def build_cost_hamiltonian_with_routes(G, routes, node_to_qubit, cycle_time=2):

    half_cycle = cycle_time / 2
    processed_edges = set()
    coeffs = []
    obs = []
    print("ПОСТРОЕНИЕ УЛУЧШЕННОГО ГАМИЛЬТОНИАНА")
    print("\n1. ШТРАФ ЗА ВРЕМЯ ПРОЕЗДА ПО МАРШРУТАМ")

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

            # <--- 2. Используем int() вместо round()
            # Нас интересует, в какой по счету полупериод прибывает машина
            half_cycles_int = int(travel_time / half_cycle)

            base_weight = (priority * volume * travel_time) / 1000.0

            if half_cycles_int % 2 == 0:
                # Прибытие на четный такт (0, 2, 4...) -> фазы должны совпадать
                weight = -base_weight
            else:
                # Прибытие на нечетный такт (1, 3, 5...) -> фазы должны быть противоположны
                weight = base_weight

            # Записываем ребро в множество (отсортированное, чтобы 1-2 и 2-1 были одним ребром)
            edge_key = tuple(sorted((light_i, light_j)))
            processed_edges.add(edge_key)

            coeffs.append(weight)
            obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))

            print(f"    Ребро {light_i}→{light_j}: вес = {weight:.3f}")

    print("\n2. ШТРАФ ЗА НАКОПЛЕНИЕ ЗАДЕРЖЕК")

    for route in routes:
        path = route['path']
        for idx, node in enumerate(path):
            if node in node_to_qubit:
                qi = node_to_qubit[node]
                position_weight = 0.02 * (idx + 1)
                coeffs.append(position_weight)
                obs.append(qml.PauliZ(qi))
                print(f"    Перекресток {node} (позиция {idx+1}): штраф {position_weight:.3f}")

    print("\n3. ШТРАФ ЗА НЕПРАВИЛЬНУЮ ФАЗУ НА ВАЖНЫХ ПЕРЕКРЕСТКАХ")

    important_nodes = [1, 2, 5, 6, 9]

    for node in important_nodes:
        if node in node_to_qubit:
            qi = node_to_qubit[node]
            weight = 0.05
            coeffs.append(weight)
            obs.append(qml.PauliZ(qi))
            print(f"    Важный перекресток {node}: штраф {weight:.3f}")

    print("\n4. ШТРАФ ЗА ВСТРЕЧНЫЕ ПОТОКИ (КРАСНЫЙ-КРАСНЫЙ)")

    opposite_edges = [(1, 5), (2, 6), (3, 7), (4, 8)]

    for u, v in G.edges():
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]

        is_opposite = (u, v) in opposite_edges or (v, u) in opposite_edges

        if is_opposite:
            weight = 0.15
            coeffs.append(weight)
            obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))
            print(f"    Встречное ребро {u}→{v}: штраф {weight:.3f}")
        else:
            weight = 0.05
            coeffs.append(weight)
            obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))

    print("\n5. БАЗОВЫЙ ШТРАФ (все ребра)")

    for u, v in G.edges():
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]
        coeffs.append(0.02)
        obs.append(qml.PauliZ(qi) @ qml.PauliZ(qj))

    print(f"ВСЕГО СЛАГАЕМЫХ В ГАМИЛЬТОНИАНЕ: {len(coeffs)}")
    H = qml.Hamiltonian(coeffs, obs)
    return H


def evaluate_route(G, route, phase_offsets, cycle_time=2):
    path = route['path']
    total_time = 0
    current_time = 0
    total_wait = 0

    print(f"\nМаршрут: {' → '.join(map(str, path))}")
    print("-" * 40)

    for i in range(len(path) - 1):
        light_current = path[i]
        light_next = path[i + 1]

        travel_time = G[light_current][light_next]['weight']

        offset_next = phase_offsets[light_next] * (cycle_time / 2)

        arrival_time = current_time + travel_time

        phase_at_arrival = int((arrival_time + offset_next) / (cycle_time / 2)) % 2

        if phase_at_arrival == 1:
            wait_time = cycle_time / 2 - ((arrival_time + offset_next) % (cycle_time / 2))
            print(f"  Перекресток {light_next}: КРАСНЫЙ, ожидание {wait_time:.1f} мин")
            total_wait += wait_time
            current_time = arrival_time + wait_time
        else:
            print(f"  Перекресток {light_next}: ЗЕЛЕНЫЙ, проезд {travel_time} мин")
            current_time = arrival_time

        total_time += travel_time

    print("-" * 40)
    print(f"  Общее время: {total_time:.1f} мин (ожидание: {total_wait:.1f} мин)")
    
    return total_time, total_wait


def optimize_traffic_with_routes(graph, routes, depth, initial_gamma, initial_beta):
    node_to_qubit, qubit_to_node, all_nodes = build_node_to_qubit_map(graph)
    num_qubits = len(all_nodes)

    print(f"\nТранспортная сеть: {num_qubits} светофоров")
    print(f"Количество маршрутов: {len(routes)}")
    print(f"Глубина QAOA: {depth}\n")

        # Используем lightning.qubit - он написан на C++ и быстрее
    dev = qml.device("lightning.qubit", wires=num_qubits)

    H = build_cost_hamiltonian_with_routes(graph, routes, node_to_qubit)

    # Добавляем diff_method="adjoint" - это спасет память!
    @qml.qnode(dev, diff_method="adjoint")
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

    # opt = qml.GradientDescentOptimizer(stepsize=0.1)
    # opt = qml.AdamOptimizer()
    # Adam справился неплохо - первый маршрут вместо 8 минут, проехал за 7,
    # но 2-й на минуту больше
    # opt = qml.AdagradOptimizer()
    # Adagrad проехал 2-й маршрут так же как и Adam, но 1-й - за 8 минут.
    # Медленно скатывался в оптимум
    opt = qml.NesterovMomentumOptimizer()
    # Нашел самое маленькое значение целевой функции, но результаты как у Adam
    # Я бы пока оставил его)


    params = init_params

    print("Запуск оптимизации")  
    history = {'cost': [], 'gamma': [], 'beta': []}
    
    for i in range(50):
        params = opt.step(cost_function, params)
        cost = cost_function(params)
        
        history['cost'].append(cost)
        history['gamma'].append(params[:depth].copy())
        history['beta'].append(params[depth:].copy())
        
        if (i + 1) % 10 == 0:
            print(f"Шаг {i + 1:3d} | Стоимость: {cost:.4f}")


    optimal_gamma = params[:depth]
    optimal_beta = params[depth:]
    final_cost = cost_function(params)

    gamma_list = [float(g) for g in optimal_gamma]
    beta_list = [float(b) for b in optimal_beta]
    final_cost_value = float(final_cost)

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
        node = qubit_to_node[qubit_idx]
        phase_offsets[node] = bit

    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print(f"Значение целевой функции: {final_cost_value:.4f}")
    print(f"Оптимальные γ: {[round(g, 4) for g in gamma_list]}")
    print(f"Оптимальные β: {[round(b, 4) for b in beta_list]}")

    print("\nФАЗОВЫЕ СМЕЩЕНИЯ:")
    for node in sorted(phase_offsets.keys()):
        phase = phase_offsets[node]
        phase_desc = "Фаза 0 (базовая)" if phase == 0 else "Фаза 1 (сдвиг)"
        print(f"  Вершина {node:3d}: {phase_desc}")

    plot_optimization_history(history)

    return phase_offsets, final_cost_value, gamma_list, beta_list


def plot_optimization_history(history):
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['cost'])
        plt.xlabel('Итерация')
        plt.ylabel('Стоимость')
        plt.title('Сходимость оптимизации')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        for i in range(len(history['gamma'][0])):
            gamma_vals = [float(g[i]) for g in history['gamma']]
            plt.plot(gamma_vals, label=f'γ_{i}')
        plt.xlabel('Итерация')
        plt.ylabel('Значение γ')
        plt.title('Параметры γ')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        for i in range(len(history['beta'][0])):
            beta_vals = [float(b[i]) for b in history['beta']]
            plt.plot(beta_vals, label=f'β_{i}')
        plt.xlabel('Итерация')
        plt.ylabel('Значение β')
        plt.title('Параметры β')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_history.png', dpi=150)
        print("\nГрафик сохранен в 'optimization_history.png'")
        plt.show()
    except Exception as e:
        print(f"Не удалось построить график: {e}")


def compare_results(G, routes, phase_offsets):
    print("СРАВНЕНИЕ С ОРИГИНАЛЬНЫМ РЕЖИМОМ (ВСЕ ФАЗЫ = 0)")
    
    phase_offsets_original = {node: 0 for node in G.nodes()}
    
    total_improved = 0
    total_original = 0
    
    for idx, route in enumerate(routes):
        path = route['path']
        print(f"\nМаршрут {idx+1}: {' → '.join(map(str, path))}")
        
        time_imp, wait_imp = evaluate_route(G, route, phase_offsets)
        time_orig, wait_orig = evaluate_route(G, route, phase_offsets_original)
        
        print(f"\n  Результаты:")
        print(f"    Улучшенный:   {time_imp:.1f} мин (ожидание {wait_imp:.1f} мин)")
        print(f"    Оригинал:     {time_orig:.1f} мин (ожидание {wait_orig:.1f} мин)")
        print(f"    Улучшение:    {time_orig - time_imp:.1f} мин ({(1 - time_imp/time_orig)*100:.1f}%)")
        
        total_improved += time_imp
        total_original += time_orig
    
    print(f"ИТОГО ПО ВСЕМ МАРШРУТАМ:")
    print(f"  Улучшенный:   {total_improved:.1f} мин")
    print(f"  Оригинал:     {total_original:.1f} мин")
    print(f"  Улучшение:    {total_original - total_improved:.1f} мин ({(1 - total_improved/total_original)*100:.1f}%)")


if __name__ == "__main__":
    G = create_traffic_graph_with_routes('traffic_graph_final.json')

    routes = define_routes(G)

    num_phases = 2
    depth = 4
    initial_gamma = [0.5, 0.3, 0.2, 0.1]
    initial_beta = [0.2, 0.4, 0.3, 0.2]
    # увеличил глубину алгоритма - выполняется дольше,
    # результат не поменялся (т.к. пример простой),
    # но для большего числа светофоров - должно быть лучше

    phase_offsets, cost, opt_gamma, opt_beta = optimize_traffic_with_routes(
        graph=G,
        routes=routes,
        depth=depth,
        initial_gamma=initial_gamma,
        initial_beta=initial_beta
    )

    print("ОЦЕНКА МАРШРУТОВ")

    for route in routes:
        time, wait = evaluate_route(G, route, phase_offsets)
        print(f"Общее время маршрута: {time:.1f} мин (ожидание: {wait:.1f} мин)")
    
    compare_results(G, routes, phase_offsets)