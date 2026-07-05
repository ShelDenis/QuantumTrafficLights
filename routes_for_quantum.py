def define_routes(G):
    routes = [
        {
            'source': 25,
            'destination': 41,
            'path': [25, 12, 40, 41],
            'priority': 1.5,
            'traffic_volume': 145
        },
        {
            'source': 2,
            'destination': 25,
            'path': [2, 11, 12, 25],
            'priority': 1.2,
            'traffic_volume': 124
        },
        {
            'source': 17,
            'destination': 2,
            'path': [17, 18, 14, 12, 11, 2],
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
            'source': 13,
            'destination': 31,
            'path': [13, 12, 14, 18, 31],
            'priority': 1.1,
            'traffic_volume': 113
        },
    ]
    return routes
