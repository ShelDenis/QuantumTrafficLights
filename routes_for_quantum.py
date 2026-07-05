# Оптимальные маршруты для квантовой оптимизации
# Найдено эволюционным алгоритмом
# Уникальных вершин: 11

def define_routes(G):
    routes = [
        {
            'source': 2,
            'destination': 25,
            'path': [2, 11, 12, 25],
            'priority': 1.2,
            'traffic_volume': 124
        },
        {
            'source': 13,
            'destination': 6,
            'path': [13, 12, 11, 2, 6],
            'priority': 1.2,
            'traffic_volume': 118
        },
        {
            'source': 12,
            'destination': 5,
            'path': [12, 11, 2, 5],
            'priority': 1.3,
            'traffic_volume': 126
        },
        {
            'source': 5,
            'destination': 2,
            'path': [5, 3, 1, 2],
            'priority': 1.2,
            'traffic_volume': 118
        },
        {
            'source': 17,
            'destination': 40,
            'path': [17, 6, 2, 11, 12, 40],
            'priority': 1.3,
            'traffic_volume': 130
        },
    ]
    return routes
