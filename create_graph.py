import json
import time
import requests
from datetime import datetime, timedelta
import re

# Ваш API ключ TomTom
API_KEY = ""

# Настройки задержек для API
TOMTOM_DELAY = 0.3  # секунды между запросами к TomTom
OSM_DELAY = 10  # секунды между запросами к Overpass API
MAX_OSM_REQUESTS_PER_MINUTE = 30  # лимит Overpass API
MAX_RETRIES = 3  # максимальное количество повторных попыток

osm_cache = {}

class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests_timestamps = []
        self.total_requests = 0

    def wait_if_needed(self):
        current_time = datetime.now()

        self.requests_timestamps = [t for t in self.requests_timestamps
                                    if (current_time - t) < timedelta(minutes=1)]

        # Проверяем лимит
        if len(self.requests_timestamps) >= self.max_requests:
            # Ждем до освобождения слота
            oldest = min(self.requests_timestamps)
            wait_time = 60 - (current_time - oldest).total_seconds()
            if wait_time > 0:
                time.sleep(wait_time)

        self.requests_timestamps.append(datetime.now())
        self.total_requests += 1

osm_limiter = RateLimiter(MAX_OSM_REQUESTS_PER_MINUTE)

def parse_frc(frc_value):
    if frc_value is None:
        return None

    if isinstance(frc_value, (int, float)):
        return int(frc_value)

    if isinstance(frc_value, str):
        match = re.search(r'(\d+)', frc_value)
        if match:
            return int(match.group(1))
        return int(frc_value)


    return None


def get_road_importance_from_osm(lat, lon, radius=50):
    overpass_url = "https://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    way(around:{radius},{lat},{lon})["highway"];
    out body;
    """

    for attempt in range(MAX_RETRIES):
        osm_limiter.wait_if_needed()

        print(f"  OSM запрос {osm_limiter.total_requests} (попытка {attempt + 1})...")

        if attempt > 0:
            time.sleep(OSM_DELAY * 2)

        response = requests.post(
            overpass_url,
            data={"data": overpass_query},
            timeout=30,
            headers={
                "User-Agent": "TrafficAnalyzer/1.0 (mikhail05@list.ru)"
            }
        )

        if response.status_code == 200:
            data = response.json()

            if data['elements']:
                roads_info = []
                for element in data['elements']:
                    if 'tags' in element and 'highway' in element['tags']:
                        highway_type = element['tags']['highway']

                        importance_scores = {
                            'motorway': 10,
                            'motorway_link': 9,
                            'trunk': 9,
                            'trunk_link': 8,
                            'primary': 8,
                            'primary_link': 7,
                            'secondary': 6,
                            'secondary_link': 5,
                            'tertiary': 5,
                            'tertiary_link': 4,
                            'unclassified': 3,
                            'residential': 2,
                            'service': 1,
                            'living_street': 1,
                            'pedestrian': 0,
                            'track': 0,
                            'road': 1
                        }

                        importance = importance_scores.get(highway_type, 0)

                        road_info = {
                            'highway_type': highway_type,
                            'importance': importance,
                            'name': element['tags'].get('name', 'Без названия'),
                            'lanes': element['tags'].get('lanes', 'не указано'),
                            'maxspeed': element['tags'].get('maxspeed', 'не указано'),
                            'oneway': element['tags'].get('oneway', 'no')
                        }
                        roads_info.append(road_info)

                if roads_info:
                    most_important = max(roads_info, key=lambda x: x['importance'])
                    return most_important

            return None
    return None


def calculate_road_importance_weight(osm_info, tomtom_frc=None):
    importance = 1.0

    if osm_info:
        osm_weight = 0.3 + (osm_info['importance'] * 0.07)
        importance *= osm_weight

    if tomtom_frc is not None:
        frc_num = parse_frc(tomtom_frc)
        if frc_num is not None:
            frc_weight = 1.0 - (frc_num * 0.075)
            importance *= frc_weight

    return min(importance, 2.0)


def calculate_improved_fitness(tomtom_data, osm_info=None):
    frc_raw = tomtom_data.get('frc')
    frc = parse_frc(frc_raw)

    if frc is None:
        frc_score = 0.5
    else:
        frc_score = max(0.1, 1.0 - frc / 8.0)

    osm_score = 0.5
    if osm_info:
        osm_score = osm_info.get("importance", 3) / 10.0

    highway = osm_info.get("highway_type") if osm_info else None

    highway_boost = {
        "motorway": 1.5,
        "trunk": 1.4,
        "primary": 1.3,
        "secondary": 1.1,
        "tertiary": 1.0,
        "residential": 0.6,
        "service": 0.3,
        "track": 0.2,
        None: 0.8
    }.get(highway, 0.8)

    lanes = osm_info.get("lanes") if osm_info else None
    try:
        lanes = int(lanes) if lanes not in [None, "не указано"] else 1
    except:
        lanes = 1

    lanes_score = min(lanes / 4.0, 1.5)

    structural_importance = (
        frc_score * 0.4 +
        osm_score * 0.4 +
        highway_boost * 0.2
    )

    importance = structural_importance * lanes_score

    final_score = min(max(importance * 100, 0), 100)

    return {
        "frc_score": round(frc_score, 3),
        "osm_score": round(osm_score, 3),
        "highway_boost": highway_boost,
        "lanes_score": round(lanes_score, 3),
        "raw_importance": round(importance, 3),
        "normalized_fitness": round(final_score, 2)
    }

raw_data = [
    [1, 55.044135, 73.319328, 55.045445, 73.314702, 55.044700, 73.317372],
    [2, 55.044135, 73.319328, 55.042743, 73.324049, 55.043430, 73.321710],
    [3, 55.042743, 73.324049, 55.040757, 73.335192, 55.041083, 73.329226],
    [4, 55.040757, 73.335192, 55.048688, 73.328771, 55.045674, 73.331239],
    [5, 55.042743, 73.324049, 55.048688, 73.328771, 55.046990, 73.325998],
    [6, 55.045445, 73.314702, 55.046763, 73.306634, 55.046171, 73.310383],
    [7, 55.046763, 73.306634, 55.047494, 73.296552, 55.047130, 73.301781],
    [8, 55.047494, 73.296552, 55.051399, 73.299174, 55.049668, 73.298042],
    [9, 55.046763, 73.306634, 55.051399, 73.299174, 55.049213, 73.302549],
    [10, 55.051399, 73.299174, 55.052795, 73.299959, 55.052096, 73.299549],
    [11, 55.048688, 73.328771, 55.052795, 73.299959, 55.052716, 73.325285],
    [12, 55.047494, 73.296552, 55.042533, 73.261202, 55.047282, 73.281364],
    [13, 55.045445, 73.314702, 55.038486, 73.311382, 55.041955, 73.312849],
    [14, 55.038486, 73.311382, 55.036963, 73.310689, 55.037893, 73.311114],
    [15, 55.036963, 73.310689, 55.018520, 73.302650, 55.025753, 73.305645],
    [16, 55.036963, 73.310689, 55.037474, 73.308759, 55.037212, 73.309673],
    [17, 55.037474, 73.308759, 55.037981, 73.306861, 55.037725, 73.307830],
    [18, 55.037981, 73.306861, 55.039932, 73.299995, 55.038716, 73.304171],
    [19, 55.039932, 73.299995, 55.041885, 73.293204, 55.040683, 73.297307],
    [20, 55.046763, 73.306634, 55.041885, 73.293204, 55.044392, 73.294693],
    [21, 55.041885, 73.293204, 55.036786, 73.289914, 55.039363, 73.291585],
    [22, 55.037474, 73.308759, 55.036786, 73.289914, 55.032210, 73.303365],
    [23, 55.036786, 73.289914, 55.035012, 73.288841, 55.035813, 73.289289],
    [24, 55.035012, 73.288841, 55.032995, 73.287370, 55.034036, 73.288162],
    [25, 55.032995, 73.287370, 55.029206, 73.285013, 55.030576, 73.285948],
    [26, 55.029206, 73.285013, 55.026873, 73.291269, 55.027775, 73.289003],
    [27, 55.026873, 73.291269, 55.025681, 73.294540, 55.026272, 73.292940],
    [28, 55.025681, 73.294540, 55.024876, 73.297016, 55.023820, 73.299500],
    [29, 55.024876, 73.297016, 55.023207, 73.301299, 55.023744, 73.299679],
    [30, 55.023207, 73.301299, 55.018520, 73.302650, 55.022169, 73.303481],
    [31, 55.048688, 73.328771, 55.045445, 73.314702, 55.051304, 73.317234],
    [32, 55.042533, 73.261202, 55.041966, 73.261538, 55.042205, 73.261245],
    [33, 55.041966, 73.261538, 55.038066, 73.264618, 55.039696, 73.263255],
    [34, 55.041885, 73.293204, 55.042219, 73.287822, 55.042211, 73.290082],
    [35, 55.042219, 73.287822, 55.042171, 73.285030, 55.042192, 73.287223],
    [36, 55.042171, 73.285030, 55.042025, 73.279916, 55.042113, 73.282202],
    [37, 55.042025, 73.279916, 55.038066, 73.264618, 55.039186, 73.269258],
    [38, 55.038066, 73.264618, 55.034618, 73.267445, 55.036255, 73.266075],
    [39, 55.036786, 73.289914, 55.034618, 73.267445, 55.036671, 73.278371],
    [40, 55.034618, 73.267445, 55.032232, 73.269351, 55.032884, 73.268869],
    [41, 55.032995, 73.287370, 55.032232, 73.269351, 55.033173, 73.272906],
    [42, 55.032232, 73.269351, 55.029233, 73.271285, 55.030779, 73.270388],
    [43, 55.032995, 73.287370, 55.029233, 73.271285, 55.030655, 73.276595],
    [44, 55.029233, 73.271285, 55.028995, 73.272840, 55.029025, 73.271208],
    [45, 55.028995, 73.272840, 55.029046, 73.277901, 55.029012, 73.275236],
    [46, 55.029046, 73.277901, 55.029206, 73.285013, 55.029094, 73.281200],
    [47, 55.023207, 73.301299, 55.020922, 73.307605, 55.022261, 73.303907],
    [48, 55.036963, 73.310689, 55.023207, 73.301299, 55.024422, 73.305176],
    [49, 55.018520, 73.302650, 55.006408, 73.287256, 55.014983, 73.298522],
    [50, 55.020922, 73.307605, 55.017450, 73.315929, 55.019546, 73.310800],
    [51, 55.017450, 73.315929, 55.012640, 73.331885, 55.014421, 73.325863],
    [52, 55.036963, 73.310689, 55.031814, 73.329027, 55.033112, 73.324805],
    [53, 55.031814, 73.329027, 55.029098, 73.336768, 55.030628, 73.332711],
    [54, 55.029098, 73.336768, 55.021826, 73.338918, 55.024481, 73.338217],
    [55, 55.021826, 73.338918, 55.017325, 73.339172, 55.018116, 73.339836],
    [56, 55.017325, 73.339172, 55.012640, 73.331885, 55.016941, 73.338436],
    [57, 55.040757, 73.335192, 55.033951, 73.341982, 55.035565, 73.340490],
    [58, 55.033951, 73.341982, 55.029098, 73.336768, 55.032233, 73.336322],
]

light_coords = {}
light_id_counter = 1

for row in raw_data:
    coord1 = (round(row[1], 6), round(row[2], 6))
    coord2 = (round(row[3], 6), round(row[4], 6))

    if coord1 not in light_coords:
        light_coords[coord1] = light_id_counter
        light_id_counter += 1

    if coord2 not in light_coords:
        light_coords[coord2] = light_id_counter
        light_id_counter += 1

edges = []

for idx, row in enumerate(raw_data, 1):
    coord1 = (round(row[1], 6), round(row[2], 6))
    coord2 = (round(row[3], 6), round(row[4], 6))
    mid_lat = row[5]
    mid_lon = row[6]

    light_u = light_coords[coord1]
    light_v = light_coords[coord2]

    print(f"\n[{idx}/{len(raw_data)}] Обработка ребра {light_u}→{light_v}")

    cache_key = (round(mid_lat, 4), round(mid_lon, 4))
    if cache_key not in osm_cache:
        osm_info = get_road_importance_from_osm(mid_lat, mid_lon)
        osm_cache[cache_key] = osm_info
        time.sleep(OSM_DELAY)
    else:
        osm_info = osm_cache[cache_key]
        print(f"  Использован кэш OSM")

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{mid_lat},{mid_lon}",
        "key": API_KEY,
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()["flowSegmentData"]

        road_name = data.get('roadName', 'не указано')
        frc_raw = data.get('frc', 'не указан')
        confidence = data.get('confidence', 'не указан')

        frc_parsed = parse_frc(frc_raw)

        print(f"  Дорога: {road_name}")
        print(f"  FRC: {frc_raw} → {frc_parsed}")

        tomtom_data = dict(data)
        tomtom_data['frc'] = frc_parsed

        fitness_data = calculate_improved_fitness(tomtom_data, osm_info)

        print(f"  Фитнес: {fitness_data['normalized_fitness']} ")

        edge = {
            "edge_id": idx,
            "start": light_u,
            "end": light_v,
            "midpoint": {"lat": mid_lat, "lon": mid_lon},
            "road_name": road_name,
            "current_travel_time": data["currentTravelTime"],
            "free_travel_time": data["freeFlowTravelTime"],
            "confidence": confidence,
            "frc": frc_raw,
            "frc_numeric": frc_parsed,
            "road_closed": data.get("roadClosure", False),
            "osm_type": osm_info['highway_type'] if osm_info else None,
            "osm_name": osm_info['name'] if osm_info else None,
            "osm_importance": osm_info['importance'] if osm_info else None,
            "fitness": fitness_data['normalized_fitness'],
            "current_speed": data["currentSpeed"],
            "free_speed": data["freeFlowSpeed"],
        }

        edges.append(edge)

        if idx % 10 == 0:
            with open(f"traffic_graph_partial_{idx}.json", "w", encoding="utf-8") as f:
                partial_graph = {
                    "nodes": {lid: {"id": lid, "lat": lat, "lon": lon}
                              for (lat, lon), lid in light_coords.items()},
                    "edges": edges
                }
                json.dump(partial_graph, f, ensure_ascii=False, indent=2)

    time.sleep(TOMTOM_DELAY)

graph = {
    "nodes": {},
    "edges": [],
    "metadata": {
        "created_at": datetime.now().isoformat(),
        "total_osm_requests": osm_limiter.total_requests,
        "osm_cache_hits": len(osm_cache),
        "fitness_version": "2.0_improved_with_osm"
    }
}

for (lat, lon), lid in light_coords.items():
    graph["nodes"][lid] = {
        "id": lid,
        "lat": lat,
        "lon": lon
    }

graph["edges"] = edges

with open("traffic_graph_final.json", "w", encoding="utf-8") as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)

print()
print(f"Светофоров: {len(graph['nodes'])}")
print(f"Ребер: {len(graph['edges'])}")

fitnesses = [e["fitness"] for e in edges]

print(f"\nФитнес:")
print(f"  Средний: {sum(fitnesses) / len(fitnesses):.3f}")
print(f"  Макс: {max(fitnesses):.3f}")
print(f"  Мин: {min(fitnesses):.3f}")

sorted_edges = sorted(edges, key=lambda x: x["fitness"], reverse=True)
for i, edge in enumerate(sorted_edges[:5], 1):
    print(f"  {i}. {edge['start']}→{edge['end']} "
          f"[{edge['osm_type'] or '?'}] "
          f"фитнес: {edge['fitness']:.3f}")