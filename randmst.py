#!/usr/bin/env python3
import sys
import math
import random
import matplotlib.pyplot as plt

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def unite(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


def generate_mst_weight(n, dim):
    """Creates graph and finds MST weight."""
    
    # Graph generation
    edges = []
    if dim == 0:
        # k(n) = 2 * log(n)/n
        limit = 2.0 * math.log(n, 2) / n
        for u in range(n):
            for v in range(u + 1, n):
                w = random.uniform(0., 1.)
                if w < limit:
                    edges.append((u, v, w))
    elif dim == 1:
        for u in range(n):
            for k in range(int(math.log2(n)) + 1):
                v = u ^ (1 << k)
                if v > u and v < n:
                    w = random.uniform(0., 1.)
                    if w < limit:
                        edges.append((u, v, w))
    elif dim in [2, 3, 4]:
        # k(n) = 1.25 * (log(n) / n) ^ 1/dim
        if dim == 2:
            limit = 1.25 * math.sqrt(math.log(n) / n)
        elif dim == 3:
            limit = 1.25 * (math.log(n) / n)**(1/3)
        else:
            limit = 1.25 * (math.log(n) / n)**(0.25)
            
        points = []
        for i in range(n):
            coords = [random.uniform(0, 1) for _ in range(dim)]
            points.append((i, coords))
        
        # Sort by coordinates
        points.sort(key=lambda p: p[1][0])
        
        for i in range(n):
            u_idx, u_coords = points[i]
            
            for j in range(i + 1, n):
                v_idx, v_coords = points[j]
                
                # x-distance is already too large, move on to next vertex
                if v_coords[0] - u_coords[0] > limit:
                    break
                    
                sq_dist = 0
                for k in range(dim):
                    diff = u_coords[k] - v_coords[k]
                    sq_dist += diff * diff
                
                if sq_dist < limit * limit:
                    edges.append((u_idx, v_idx, math.sqrt(sq_dist)))

    # Kruskal's algorithm
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    weight = 0
    count = 0
    
    for u, v, w in edges:
        if uf.unite(u, v):
            weight += w
            count += 1
            if count == n - 1:
                break
    if count < n - 1:
        # Bad
        return -1
    return weight


def run_single_experiment(n_points, n_trials, dimension):
    """Runs single iteration of experiment."""
    total_weight = 0
    valid_trials = 0
    
    for _ in range(n_trials):
        w = generate_mst_weight(n_points, dimension)
        if w != -1:
            total_weight += w
            valid_trials += 1
        else:
            print(f"Warning: Trial failed for N={n_points} Dim={dimension}")
        
    
    if valid_trials == 0:
        return 0
        
    return total_weight / valid_trials


def plot_experiments():
    """Auto-runs trials for all dimensions and plots the results."""
    print("Running automated experiments...")
    configs = {
        0: {"func": lambda n: 1.202, "label": "1.202", "ylim": (0.1, 2.5)},
        1: {"func": lambda n: 1.18 * (n / math.log2(n)) if n > 1 else 0, "label": r"$1.18 \cdot n / \log_2 n$"},
        2: {"func": lambda n: 0.65 * (n ** 0.5), "label": r"$0.65 \cdot \sqrt{n}$"},
        3: {"func": lambda n: 0.65 * (n ** (2/3)), "label": r"$0.65 \cdot n^{2/3}$"},
        4: {"func": lambda n: 0.69 * (n ** 0.75), "label": r"$0.69 \cdot n^{3/4}$"}
    }
    
    # n_values = [128, 256, 512, 1024, 2048, 4096] 
    n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    one_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 
                  131072, 262144]
    
    # dims = [0, 1, 2, 3, 4] 
    # results = {d: [] for d in dims}
    
    # for d in dims:
    #     print(f"Processing Dimension {d}...")
    #     if d == 1:
    #         for n in one_values:
    #             # Run 5 trials per N
    #             avg = run_single_experiment(n, 5, d)
    #             results[d].append(avg)
    #             print(f"  N={n}: {avg:.4f}")
    #         continue
    #     for n in n_values:
    #         # Run 5 trials per N
    #         avg = run_single_experiment(n, 5, d)
    #         results[d].append(avg)
    #         print(f"  N={n}: {avg:.4f}")

    # Plotting
    # plt.figure(figsize=(10, 6))
    # for d in dims:
    #     if d == 1:
    #         plt.plot(one_values, results[d], marker='o', label=f'Dim {d}')
    #     else:
    #         plt.plot(n_values, results[d], marker='o', label=f'Dim {d}')
    
    # plt.xlabel('Number of Points (n)')
    # plt.ylabel('Average MST Weight')
    # plt.title('MST Weight Scaling by Dimension')
    # plt.legend()
    # plt.grid(True)
    # plt.xscale('log')
    
    # plt.savefig('mst_scaling.png')
    # print("Plot saved to 'mst_scaling.png'")

    for d in range(5):
        print(f"Processing Dimension {d}...")
        x = one_values if d == 1 else n_values
        config = configs[d]
        y = []
        for n in x:
            avg = run_single_experiment(n, 5, d)
            y.append(avg)
            print(f"  N={n}: {avg:.4f}")

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker='o', label="Experimental")
        theory_y = [config["func"](n) for n in x]
        plt.plot(x, theory_y, linestyle='--', color='orange', 
                label=f"f(n): {config['label']}")
        plt.title(f"Dimension {d} Scaling")
        plt.xlabel("Number of Points (n)")
        plt.ylabel("Average MST Weight")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        if "ylim" in config:
            plt.ylim(*config["ylim"])
        filename = f"mst_dim_{d}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./randmst 0 numpoints numtrials dimension")
        sys.exit(1)

    mode = int(sys.argv[1])

    if mode == 1:
        # For testing
        plot_experiments()
    else:
        if len(sys.argv) != 5:
            print("Usage: ./randmst 0 numpoints numtrials dimension")
            sys.exit(1)
            
        n_points = int(sys.argv[2])
        n_trials = int(sys.argv[3])
        dimension = int(sys.argv[4])
        
        avg = run_single_experiment(n_points, n_trials, dimension)
        
        print(f"{avg:.5f} {n_points} {n_trials} {dimension}")