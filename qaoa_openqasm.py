import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from routes_for_quantum import define_routes


DEFAULT_GAMMA = [0.5, 0.3, 0.2, 0.1]
DEFAULT_BETA = [0.2, 0.4, 0.3, 0.2]


@dataclass(frozen=True)
class IsingModel:
    node_to_qubit: Dict[int, int]
    qubit_to_node: Dict[int, int]
    z_terms: Dict[int, float]
    zz_terms: Dict[Tuple[int, int], float]

    @property
    def num_qubits(self) -> int:
        return len(self.node_to_qubit)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def load_route_subgraph(graph_path: Path) -> Tuple[List[dict], Dict[Tuple[int, int], float]]:
    with graph_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    routes = define_routes(None)
    used_nodes = {node for route in routes for node in route["path"]}
    edge_weights: Dict[Tuple[int, int], float] = {}

    for edge in data["edges"]:
        u = edge["start"]
        v = edge["end"]
        if u not in used_nodes or v not in used_nodes:
            continue

        node_u = data["nodes"][str(u)]
        node_v = data["nodes"][str(v)]
        dist_km = haversine_km(
            node_u["lat"],
            node_u["lon"],
            node_v["lat"],
            node_v["lon"],
        )
        speed = edge.get("current_speed") or 35
        travel_time = max(0.1, (dist_km / speed) * 60)
        edge_weights[tuple(sorted((u, v)))] = travel_time

    return routes, edge_weights


def build_ising_model(graph_path: Path, cycle_time: float = 2.0) -> IsingModel:
    routes, edge_weights = load_route_subgraph(graph_path)
    all_nodes = sorted({node for route in routes for node in route["path"]})
    node_to_qubit = {node: idx for idx, node in enumerate(all_nodes)}
    qubit_to_node = {idx: node for node, idx in node_to_qubit.items()}

    z_terms: Dict[int, float] = defaultdict(float)
    zz_terms: Dict[Tuple[int, int], float] = defaultdict(float)
    half_cycle = cycle_time / 2

    for route in routes:
        path = route["path"]
        priority = route["priority"]
        volume = route["traffic_volume"]

        for light_i, light_j in zip(path, path[1:]):
            edge_key = tuple(sorted((light_i, light_j)))
            if edge_key not in edge_weights:
                continue

            qi = node_to_qubit[light_i]
            qj = node_to_qubit[light_j]
            travel_time = edge_weights[edge_key]
            half_cycles_int = int(travel_time / half_cycle)
            base_weight = (priority * volume * travel_time) / 1000.0
            weight = -base_weight if half_cycles_int % 2 == 0 else base_weight
            zz_terms[tuple(sorted((qi, qj)))] += weight

    for route in routes:
        for idx, node in enumerate(route["path"]):
            z_terms[node_to_qubit[node]] += 0.02 * (idx + 1)

    for node in [1, 2, 5, 6, 9]:
        if node in node_to_qubit:
            z_terms[node_to_qubit[node]] += 0.05

    opposite_edges = {(1, 5), (2, 6), (3, 7), (4, 8)}
    for u, v in edge_weights:
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]
        if (u, v) in opposite_edges or (v, u) in opposite_edges:
            zz_terms[tuple(sorted((qi, qj)))] += 0.15
        else:
            zz_terms[tuple(sorted((qi, qj)))] += 0.05

    for u, v in edge_weights:
        qi = node_to_qubit[u]
        qj = node_to_qubit[v]
        zz_terms[tuple(sorted((qi, qj)))] += 0.02

    return IsingModel(
        node_to_qubit=node_to_qubit,
        qubit_to_node=qubit_to_node,
        z_terms=dict(z_terms),
        zz_terms=dict(zz_terms),
    )


def format_angle(value: float) -> str:
    if abs(value) < 1e-14:
        value = 0.0
    return f"{value:.16g}"


def generate_qaoa_qasm(
    model: IsingModel,
    gamma: Sequence[float],
    beta: Sequence[float],
    include_measurements: bool = True,
) -> str:
    if len(gamma) != len(beta):
        raise ValueError("gamma and beta must have the same length")

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{model.num_qubits}];",
    ]
    if include_measurements:
        lines.append(f"creg c[{model.num_qubits}];")

    lines.append("")
    lines.append("// Initial uniform superposition")
    for qubit in range(model.num_qubits):
        lines.append(f"h q[{qubit}];")

    for layer, (gamma_value, beta_value) in enumerate(zip(gamma, beta)):
        lines.append("")
        lines.append(f"// QAOA layer {layer}: cost unitary")

        for qubit, coeff in sorted(model.z_terms.items()):
            angle = 2 * gamma_value * coeff
            lines.append(f"rz({format_angle(angle)}) q[{qubit}];")

        for (qi, qj), coeff in sorted(model.zz_terms.items()):
            angle = 2 * gamma_value * coeff
            lines.append(f"cx q[{qi}],q[{qj}];")
            lines.append(f"rz({format_angle(angle)}) q[{qj}];")
            lines.append(f"cx q[{qi}],q[{qj}];")

        lines.append(f"// QAOA layer {layer}: mixer unitary")
        mixer_angle = 2 * beta_value
        for qubit in range(model.num_qubits):
            lines.append(f"rx({format_angle(mixer_angle)}) q[{qubit}];")

    if include_measurements:
        lines.append("")
        for qubit in range(model.num_qubits):
            lines.append(f"measure q[{qubit}] -> c[{qubit}];")

    return "\n".join(lines) + "\n"


def bit_to_z(bit: int) -> int:
    return 1 if bit == 0 else -1


def energy_for_index(index: int, model: IsingModel) -> float:
    energy = 0.0
    for qubit, coeff in model.z_terms.items():
        energy += coeff * bit_to_z((index >> qubit) & 1)
    for (qi, qj), coeff in model.zz_terms.items():
        energy += coeff * bit_to_z((index >> qi) & 1) * bit_to_z((index >> qj) & 1)
    return energy


def energy_for_bitstring(bitstring: str, model: IsingModel) -> float:
    # Qiskit count keys are printed as c[n-1]...c[0], so reverse to qubit order.
    bits = list(reversed(bitstring.replace(" ", "")))
    energy = 0.0
    for qubit, coeff in model.z_terms.items():
        energy += coeff * bit_to_z(int(bits[qubit]))
    for (qi, qj), coeff in model.zz_terms.items():
        energy += coeff * bit_to_z(int(bits[qi])) * bit_to_z(int(bits[qj]))
    return energy


def expected_energy_from_counts(counts: Dict[str, int], model: IsingModel) -> float:
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("Cannot estimate energy from empty counts")
    return sum(energy_for_bitstring(bits, model) * count for bits, count in counts.items()) / shots


def most_likely_phase_offsets(counts: Dict[str, int], model: IsingModel) -> Dict[int, int]:
    best_bits = max(counts.items(), key=lambda item: item[1])[0]
    bits = list(reversed(best_bits.replace(" ", "")))
    return {
        model.qubit_to_node[qubit]: int(bits[qubit])
        for qubit in range(model.num_qubits)
    }


def apply_rx_to_qubit(
    state: List[complex],
    qubit: int,
    beta_value: float,
) -> None:
    cos_value = math.cos(beta_value)
    sin_value = math.sin(beta_value)
    step = 1 << qubit
    period = step << 1

    for base in range(0, len(state), period):
        for offset in range(step):
            i0 = base + offset
            i1 = i0 + step
            amp0 = state[i0]
            amp1 = state[i1]
            state[i0] = cos_value * amp0 - 1j * sin_value * amp1
            state[i1] = -1j * sin_value * amp0 + cos_value * amp1


class StatevectorBackend:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def run(
        self,
        model: IsingModel,
        gamma: Sequence[float],
        beta: Sequence[float],
        shots: int,
    ) -> Dict[str, int]:
        size = 1 << model.num_qubits
        amplitude = 1 / math.sqrt(size)
        state = [complex(amplitude, 0.0) for _ in range(size)]
        energies = [energy_for_index(index, model) for index in range(size)]

        for gamma_value, beta_value in zip(gamma, beta):
            for index, energy in enumerate(energies):
                phase = -gamma_value * energy
                state[index] *= complex(math.cos(phase), math.sin(phase))
            for qubit in range(model.num_qubits):
                apply_rx_to_qubit(state, qubit, beta_value)

        probabilities = [abs(amplitude) ** 2 for amplitude in state]
        return sample_probabilities(probabilities, shots, model.num_qubits, self.random)


class QiskitBackend:
    def __init__(self, backend=None, seed: Optional[int] = None):
        self.backend = backend
        self.seed = seed

    def run_qasm(self, qasm: str, shots: int) -> Dict[str, int]:
        try:
            from qiskit import QuantumCircuit, transpile
        except ImportError as exc:
            raise RuntimeError("Qiskit is required for QiskitBackend") from exc

        circuit = QuantumCircuit.from_qasm_str(qasm)
        backend = self.backend or self._default_backend()
        run_circuit = transpile(circuit, backend)
        job = backend.run(run_circuit, shots=shots)
        return dict(job.result().get_counts())

    def _default_backend(self):
        try:
            from qiskit_aer import AerSimulator

            return AerSimulator(seed_simulator=self.seed)
        except ImportError as exc:
            raise RuntimeError(
                "qiskit-aer is not installed. Install qiskit-aer or pass a backend object."
            ) from exc


def sample_probabilities(
    probabilities: Sequence[float],
    shots: int,
    num_qubits: int,
    rng: random.Random,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    cumulative = []
    total = 0.0
    for probability in probabilities:
        total += probability
        cumulative.append(total)

    for _ in range(shots):
        draw = rng.random() * total
        lo = 0
        hi = len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < draw:
                lo = mid + 1
            else:
                hi = mid
        bitstring = format(lo, f"0{num_qubits}b")
        counts[bitstring] += 1

    return dict(counts)


def run_backend(
    model: IsingModel,
    gamma: Sequence[float],
    beta: Sequence[float],
    shots: int,
    backend_name: str,
    seed: Optional[int],
) -> Dict[str, int]:
    if backend_name == "statevector":
        return StatevectorBackend(seed=seed).run(model, gamma, beta, shots)
    if backend_name == "qiskit-aer":
        qasm = generate_qaoa_qasm(model, gamma, beta, include_measurements=True)
        return QiskitBackend(seed=seed).run_qasm(qasm, shots)
    raise ValueError(f"Unsupported backend: {backend_name}")


def spsa_optimize(
    model: IsingModel,
    depth: int,
    iterations: int,
    shots: int,
    backend_name: str,
    seed: Optional[int],
    initial_gamma: Optional[Sequence[float]] = None,
    initial_beta: Optional[Sequence[float]] = None,
) -> Tuple[List[float], List[float], float, Dict[str, int]]:
    rng = random.Random(seed)
    gamma = normalize_parameters(initial_gamma, DEFAULT_GAMMA, depth)
    beta = normalize_parameters(initial_beta, DEFAULT_BETA, depth)

    theta = gamma + beta
    best_theta = list(theta)
    best_counts = run_backend(model, gamma, beta, shots, backend_name, seed)
    best_energy = expected_energy_from_counts(best_counts, model)

    a = 0.15
    c = 0.1
    alpha = 0.602
    gamma_decay = 0.101

    for iteration in range(iterations):
        ak = a / ((iteration + 1) ** alpha)
        ck = c / ((iteration + 1) ** gamma_decay)
        delta = [1 if rng.random() < 0.5 else -1 for _ in theta]
        theta_plus = [value + ck * sign for value, sign in zip(theta, delta)]
        theta_minus = [value - ck * sign for value, sign in zip(theta, delta)]

        counts_plus = run_backend(
            model,
            theta_plus[:depth],
            theta_plus[depth:],
            shots,
            backend_name,
            None if seed is None else seed + 10_000 + iteration,
        )
        counts_minus = run_backend(
            model,
            theta_minus[:depth],
            theta_minus[depth:],
            shots,
            backend_name,
            None if seed is None else seed + 20_000 + iteration,
        )
        energy_plus = expected_energy_from_counts(counts_plus, model)
        energy_minus = expected_energy_from_counts(counts_minus, model)

        gradient = [
            (energy_plus - energy_minus) / (2 * ck * sign)
            for sign in delta
        ]
        theta = [value - ak * grad for value, grad in zip(theta, gradient)]

        counts = run_backend(
            model,
            theta[:depth],
            theta[depth:],
            shots,
            backend_name,
            None if seed is None else seed + 30_000 + iteration,
        )
        energy = expected_energy_from_counts(counts, model)

        if energy < best_energy:
            best_energy = energy
            best_theta = list(theta)
            best_counts = counts

        print(
            f"iteration={iteration + 1:03d} "
            f"energy={energy:.6f} best={best_energy:.6f}"
        )

    return best_theta[:depth], best_theta[depth:], best_energy, best_counts


def write_metadata(path: Path, model: IsingModel) -> None:
    payload = {
        "node_to_qubit": model.node_to_qubit,
        "z_terms": {str(key): value for key, value in sorted(model.z_terms.items())},
        "zz_terms": {
            f"{qi},{qj}": value
            for (qi, qj), value in sorted(model.zz_terms.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def normalize_parameters(
    values: Optional[Sequence[float]],
    defaults: Sequence[float],
    depth: int,
) -> List[float]:
    params = list(values or defaults[:depth])
    while len(params) < depth:
        params.append(defaults[-1])
    return params[:depth]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenQASM 2.0 QAOA circuits and run a hybrid loop."
    )
    parser.add_argument("--graph", default="traffic_graph_final.json")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backend", choices=["statevector", "qiskit-aer"], default="statevector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", help="Comma-separated initial gamma values")
    parser.add_argument("--beta", help="Comma-separated initial beta values")
    parser.add_argument("--qasm-out", default="qaoa_traffic_lights.qasm")
    parser.add_argument("--metadata-out", default="qaoa_ising_model.json")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    model = build_ising_model(Path(args.graph))
    initial_gamma = parse_float_list(args.gamma)
    initial_beta = parse_float_list(args.beta)

    if args.no_optimize:
        gamma = normalize_parameters(initial_gamma, DEFAULT_GAMMA, args.depth)
        beta = normalize_parameters(initial_beta, DEFAULT_BETA, args.depth)
        counts = run_backend(model, gamma, beta, args.shots, args.backend, args.seed)
        energy = expected_energy_from_counts(counts, model)
    else:
        gamma, beta, energy, counts = spsa_optimize(
            model=model,
            depth=args.depth,
            iterations=args.iterations,
            shots=args.shots,
            backend_name=args.backend,
            seed=args.seed,
            initial_gamma=initial_gamma,
            initial_beta=initial_beta,
        )

    qasm = generate_qaoa_qasm(model, gamma, beta, include_measurements=True)
    Path(args.qasm_out).write_text(qasm, encoding="utf-8")
    write_metadata(Path(args.metadata_out), model)
    phase_offsets = most_likely_phase_offsets(counts, model)

    print("")
    print(f"qubits={model.num_qubits}")
    print(f"z_terms={len(model.z_terms)} zz_terms={len(model.zz_terms)}")
    print(f"best_energy={energy:.6f}")
    print(f"gamma={[round(value, 6) for value in gamma]}")
    print(f"beta={[round(value, 6) for value in beta]}")
    print(f"phase_offsets={dict(sorted(phase_offsets.items()))}")
    print(f"qasm_written={args.qasm_out}")
    print(f"metadata_written={args.metadata_out}")


if __name__ == "__main__":
    main()
