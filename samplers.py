import random
from functools import lru_cache
from collections import defaultdict

__all__ = ("sample_uniform_equibucket", "sample_approx_2d")

def _histogram_except_last(
        bucket_sizes: list[int],
        last_picked: int,
        common_size: int
    ) -> tuple[int, ...]:
    """
    Build a histogram of sizes for all buckets except the last-picked one.

    Args:
        bucket_sizes: List of current counts for each bucket.
        last_picked: Index of the last-picked bucket (0-based), or -1 if none.
        common_size: The initial maximum size of any bucket (defines histogram length).

    Returns:
        A tuple hist where hist[c] is the number of other buckets with size c.
    """
    hist = [0] * (common_size + 1)
    for idx, size in enumerate(bucket_sizes):
        if idx == last_picked:
            continue
        hist[size] += 1
    return tuple(hist)


@lru_cache(None)
def _count_completions(
        hist: tuple[int, ...],
        last_size: int
    ) -> int:
    """
    DP: count how many valid sequences remain from state (hist, last_size).

    Args:
        hist: Histogram tuple of length k+1 (k=common_size), where hist[c]
            counts the other buckets with exactly c items left.
        last_size: Number of items left in the last-picked bucket.

    Returns:
        Total number of adjacency-free completions.
    """
    # Base: no items left anywhere
    if last_size == 0 and all(count == 0 for count in hist[1:]):
        return 1

    total = 0
    k = len(hist) - 1
    # Try picking any other bucket of size c > 0
    for size in range(1, k + 1):
        count = hist[size]
        if count == 0:
            continue
        # Remove one bucket of this size, add back the last bucket
        new_hist = list(hist)
        new_hist[size] -= 1
        new_hist[last_size] += 1
        total += count * _count_completions(tuple(new_hist), size - 1)
    return total


def sample_uniform_equibucket(
        bucket_count: int,
        bucket_common_size: int
    ) -> list[int]:
    """
    Uniformly sample an ordering of bucket picks with no two adjacent from the same bucket.

    Uses a compressed DP (histogram of counts) for speed when many buckets have equal sizes.

    Args:
        bucket_count: Number of buckets (n > 2).
        bucket_common_size: Initial size k of each bucket.

    Returns:
        A list of bucket indices (0-based) representing the sampled order.
    """
    # Initialize buckets and first pick uniformly
    bucket_sizes = [bucket_common_size] * bucket_count
    first = random.randrange(bucket_count)
    bucket_sizes[first] -= 1
    ordering = [first]
    last_picked = first

    # Continue until all buckets empty
    while any(size > 0 for size in bucket_sizes):
        # Build histogram of other buckets
        hist = _histogram_except_last(bucket_sizes, last_picked, bucket_common_size)
        last_size = bucket_sizes[last_picked]

        candidates: list[int] = []
        weights: list[int] = []
        for idx, size in enumerate(bucket_sizes):
            if idx == last_picked or size == 0:
                continue
            # Compute weight via DP
            new_hist = list(hist)
            new_hist[size] -= 1
            new_hist[last_size] += 1
            w = _count_completions(tuple(new_hist), size - 1)
            candidates.append(idx)
            weights.append(w)

        # Draw next bucket by weight
        pick = random.choices(candidates, weights=weights, k=1)[0]
        ordering.append(pick)
        bucket_sizes[pick] -= 1
        last_picked = pick

    return ordering

def sample_approx_2d(
    bucket_count: int,
    other_count: int,
    repair_rounds: int = 10000
) -> list[tuple[int, int]]:
    """
    Approximate but uniqueness-preserving sampling of (bucket, performer) sequence under:
      - no two adjacent main buckets the same (exact via 1-D sampler)
      - each (bucket, performer) appears exactly once
      - locally reduce performer adjacency within each bucket (approximate)

    Args:
        bucket_count: number of main buckets (B)
        other_count: number of performers per bucket (O)
        repair_rounds: max failed swap attempts in local repair per conflict

    Returns:
        List of (bucket, performer) tuples of length B * O.
    """
    # Phase 1: exact main-bucket sequence (no repeats)
    total_per_bucket = other_count  # common_size = 1
    bucket_seq = sample_uniform_equibucket(bucket_count, total_per_bucket)
    N = len(bucket_seq)

    # Phase 2: initial unique performer assignment per bucket
    per_bucket_slots = defaultdict(list)
    for pos, b in enumerate(bucket_seq):
        per_bucket_slots[b].append(pos)

    performer_at: List[int] = [None] * N  # type: ignore
    for b, slots in per_bucket_slots.items():
        pool = list(range(other_count))
        random.shuffle(pool)
        for pos, o in zip(slots, pool):
            performer_at[pos] = o

    # Phase 3: local-repair within each bucket to reduce performer adjacency
    def find_conflicts() -> list[int]:
        return [i for i in range(N - 1) if performer_at[i] == performer_at[i + 1]]

    conflicts = find_conflicts()
    attempts = 0
    while conflicts and attempts < repair_rounds:
        i = random.choice(conflicts)
        # Only swap within the same bucket to preserve uniqueness
        b1 = bucket_seq[i + 1]
        slots = per_bucket_slots[b1]
        # choose a different position j in the same bucket
        candidates = [p for p in slots if p != i + 1]
        if not candidates:
            break
        j = random.choice(candidates)

        orig_i1 = performer_at[i + 1]
        orig_j = performer_at[j]
        performer_at[i + 1], performer_at[j] = orig_j, orig_i1

        new_conflicts = find_conflicts()
        if len(new_conflicts) < len(conflicts):
            conflicts = new_conflicts
        else:
            # undo swap
            performer_at[i + 1], performer_at[j] = orig_i1, orig_j
            attempts += 1

    return list(zip(bucket_seq, performer_at))

def main():
    # test cases: (bucket_count, other_count, max_performer_conflicts)
    cases = [(3, 2, 0), (4, 3, 1), (2, 5, 2)]
    for bc, oc, maxc in cases:
        random.seed(0)
        seq = sample_approx_2d(bucket_count=bc, other_count=oc)
        N = bc * oc
        # length and uniqueness
        assert len(seq) == N, f"Expected length {N}, got {len(seq)}"
        expected = {(b, o) for b in range(bc) for o in range(oc)}
        assert set(seq) == expected, "Missing or duplicate (bucket, performer) pairs"
        # no-repeat on main bucket
        assert all(seq[i][0] != seq[i+1][0] for i in range(N-1)), "Main-bucket adjacency detected"
        # approximate constraint on performer
        conflicts = sum(1 for i in range(N-1) if seq[i][1] == seq[i+1][1])
        assert conflicts <= maxc, f"Too many performer conflicts ({conflicts} > {maxc})"
    print("All tests passed.")

if __name__ == "__main__":
    main()

