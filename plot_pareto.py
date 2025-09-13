# plot_pareto.py
import sys, csv, matplotlib.pyplot as plt

def load_csv(path):
    rows=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for d in r:
            rows.append({k: float(v) for k,v in d.items()})
    return rows

def main(cand_csv, pareto_csv=None):
    cand = load_csv(cand_csv)
    px = [d["power"] for d in cand]
    py = [d["delay"] for d in cand]

    plt.figure()
    plt.scatter(px, py, s=18, alpha=0.5, label="Candidates")
    if pareto_csv:
        par = load_csv(pareto_csv)
        if par:
            ppx = [d["power"] for d in par]
            ppy = [d["delay"] for d in par]
            plt.plot(ppx, ppy, lw=2, label="Pareto")
    plt.xlabel("Power (lower better)")
    plt.ylabel("Delay (lower better)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pareto.py runs/<timestamp>/candidates.csv [runs/<timestamp>/pareto.csv]")
        sys.exit(1)
    cand = sys.argv[1]
    par = sys.argv[2] if len(sys.argv) > 2 else None
    main(cand, par)
