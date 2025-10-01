#pareto.py
def get_pareto(points):
    """
    points: list of dicts with 'power' and 'delay' to MINIMIZE.
    Returns nondominated subset sorted by power (asc) with strictly improving delay.
    """
    P = sorted(points, key=lambda d: (d["power"], d["delay"]))
    front, best_d = [], float("inf")
    for p in P:
        if p["delay"] < best_d:
            front.append(p)
            best_d = p["delay"]
    return front
