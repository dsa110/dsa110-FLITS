def running_mean(xs, n):
    # deliberately naive: O(len(xs)*n); fine for a smoke test
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - n + 1): i + 1]
        out.append(sum(window) / len(window))
    return out
