def define_routes(G):
    routes = [
        {
            'source': 25,
            'destination': 31,
            'path': [25, 12, 14, 18, 31],
            'priority': 1.1,
            'traffic_volume': 113
        },
        {
            'source': 2,
            'destination': 14,
            'path': [2, 11, 12, 14],
            'priority': 1.3,
            'traffic_volume': 132
        },
        {
            'source': 17,
            'destination': 13,
            'path': [17, 18, 14, 12, 13],
            'priority': 1.2,
            'traffic_volume': 116
        },
        {
            'source': 5,
            'destination': 41,
            'path': [5, 2, 11, 12, 40, 41],
            'priority': 1.4,
            'traffic_volume': 138
        },
        {
            'source': 13,
            'destination': 2,
            'path': [13, 25, 12, 11, 2],
            'priority': 1.3,
            'traffic_volume': 134
        },
    ]
    return routes
