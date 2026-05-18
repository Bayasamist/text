/*
 * IFN664 Advanced Algorithms — Group Project
 * Minimum-Size Text Reconstruction from Overlapping Fragments
 *
 * Authors: Group 7
 *
 * Compile:  gcc -O2 -std=gnu11 -Wall -Wextra -o text_reconstruction text_reconstruction.c
 * Usage:    ./text_reconstruction <input_file>
 *           ./text_reconstruction -          (read from stdin)
 *           Press Ctrl+C to stop early and keep the best solution found so far.
 *
 * Algorithm overview:
 *   1. Read fragments from input file (one per non-blank line).
 *   2. Preprocess: remove exact duplicates, then remove any fragment that is
 *      a substring of another — safe because any superstring covering the
 *      longer fragment already covers the shorter one.
 *   3. Build pairwise overlap matrix: ov[i][j] = length of the longest suffix
 *      of fragment i that equals a prefix of fragment j.
 *   4. Greedy solver (Phase 1): repeatedly merge the tail→head pair with the
 *      greatest overlap. O(n^2) after the matrix is built. Gives a correct
 *      solution immediately, though not always optimal.
 *   5. Held-Karp exact DP (Phase 2): bitmask DP over all subsets, equivalent
 *      to solving TSP on the overlap graph. Finds the provably shortest
 *      superstring. O(2^n · n^2) time, O(2^n · n) space. Skipped if n exceeds
 *      MAX_EXACT to avoid out-of-memory conditions on large inputs.
 *   6. SIGINT handler: Ctrl+C at any point prints the best solution found so
 *      far and exits cleanly — useful when Phase 2 is running slowly.
 *   7. Verification: confirms every original fragment appears in the output.
 *
 * Correctness argument:
 *   Preprocessing preserves correctness: dropping a fragment that is a
 *   substring of another does not change the set of valid superstrings.
 *   The greedy solution is always correct (every fragment is visited once).
 *   The exact DP solution is optimal by exhaustive search over all orderings.
 *
 * Complexity summary (n = fragment count after preprocessing, L = max length):
 *   Preprocessing      : O(n^2 · L)    — strstr is O(L) via naive search
 *   Overlap matrix     : O(n^2 · L^2)  — O(L^2) per pair (suffix-prefix scan)
 *   Greedy             : O(n^2)        — n-1 merge rounds, each O(n^2) scan
 *   Held-Karp exact DP : O(2^n · n^2) time,  O(2^n · n) space
 */

/*
 * _POSIX_C_SOURCE 200809L enables POSIX.1-2008 features used here:
 *   - struct sigaction / sigaction()  (POSIX signal handling)
 *   - getline()                       (POSIX.1-2008 I/O)
 * On non-POSIX systems (e.g. Windows/MSVC) these are unavailable;
 * the code is intended to be compiled with gcc/clang on Linux as
 * required by the assignment specification.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <limits.h>


#define MAX_EXACT 20


typedef struct {
    char **items;    /* heap-allocated array of heap-allocated strings */
    int    count;    /* number of fragments currently stored           */
    int    capacity; /* allocated slots in items[]                     */
} FragmentArray;


static volatile sig_atomic_t interrupted   = 0;
static char                 *best_solution = NULL;
static int                   best_length   = INT_MAX;
static bool                  is_optimal    = false;

static void handle_sigint(int sig) {
    (void)sig;       /* suppress unused-parameter warning */
    interrupted = 1;
}


static void fa_init(FragmentArray *fa) {
    fa->items = NULL;
    fa->count = 0;
    fa->capacity = 0;
}

static void fa_free(FragmentArray *fa) {
    if (fa->items) {
        for (int i = 0; i < fa->count; i++)
            free(fa->items[i]);
        free(fa->items);
    }
    fa_init(fa);
}

/* Appends a copy of 'text' to fa, growing the backing array as needed. */
static void fa_append(FragmentArray *fa, const char *text) {
    if (fa->count == fa->capacity) {
        int nc = (fa->capacity == 0) ? 8 : fa->capacity * 2;
        char **ni = realloc(fa->items, nc * sizeof(char *));
        if (!ni) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
        fa->items    = ni;
        fa->capacity = nc;
    }
    char *copy = strdup(text);
    if (!copy) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    fa->items[fa->count++] = copy;
}

/* Removes the element at index i by shifting subsequent elements left.
 * The removed string is freed. O(n) per call. */
static void fa_remove(FragmentArray *fa, int i) {
    free(fa->items[i]);
    for (int k = i; k < fa->count - 1; k++)
        fa->items[k] = fa->items[k + 1];
    fa->count--;
}

/* Deep-copies src into dst (dst must be uninitialised or empty). */
static void fa_copy(const FragmentArray *src, FragmentArray *dst) {
    fa_init(dst);
    for (int i = 0; i < src->count; i++)
        fa_append(dst, src->items[i]);
}

/* Strips a trailing '\n' or '\r\n' in-place. */
static void trim_newline(char *s) {
    size_t l = strlen(s);
    while (l > 0 && (s[l - 1] == '\n' || s[l - 1] == '\r'))
        s[--l] = '\0';
}

/* Reads one fragment per non-blank line.
 * Pass "-" to read from stdin instead of a file. */
static bool read_fragments(const char *filename, FragmentArray *fa) {
    FILE *fp = (strcmp(filename, "-") == 0) ? stdin : fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", filename);
        return false;
    }
    char  *line = NULL;
    size_t cap  = 0;
    while (getline(&line, &cap, fp) != -1) {
        trim_newline(line);
        if (line[0] != '\0')
            fa_append(fa, line);
    }
    free(line);
    if (fp != stdin) fclose(fp);
    return true;
}


static void remove_duplicates(FragmentArray *fa) {
    for (int i = 0; i < fa->count; i++) {
        int j = i + 1;
        while (j < fa->count) {
            if (strcmp(fa->items[i], fa->items[j]) == 0)
                fa_remove(fa, j);   /* j now points to next element */
            else
                j++;
        }
    }
}

static void remove_contained(FragmentArray *fa) {
    int i = 0;
    while (i < fa->count) {
        bool contained = false;
        for (int j = 0; j < fa->count && !contained; j++) {
            /* fa->items[i] is a substring of fa->items[j] (and not itself) */
            if (i != j && strstr(fa->items[j], fa->items[i]) != NULL)
                contained = true;
        }
        if (contained)
            fa_remove(fa, i);   /* i now points to the next element */
        else
            i++;
    }
}

static void preprocess(FragmentArray *fa) {
    remove_duplicates(fa);
    remove_contained(fa);
}


static int compute_overlap(const char *left, const char *right) {
    int ll    = (int)strlen(left);
    int rl    = (int)strlen(right);
    int limit = ll < rl ? ll : rl;   /* overlap cannot exceed either length */

    /* Try decreasing overlap lengths; return the first (largest) match. */
    for (int k = limit; k >= 1; k--) {
        bool ok = true;
        for (int i = 0; i < k && ok; i++)
            if (left[ll - k + i] != right[i]) ok = false;
        if (ok) return k;
    }
    return 0;
}

static int **build_overlap_matrix(const FragmentArray *fa) {
    int n = fa->count;
    int **ov = malloc(n * sizeof(int *));
    if (!ov) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        ov[i] = malloc(n * sizeof(int));
        if (!ov[i]) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
        for (int j = 0; j < n; j++)
            ov[i][j] = (i == j) ? 0 : compute_overlap(fa->items[i], fa->items[j]);
    }
    return ov;
}

static void free_overlap_matrix(int **ov, int n) {
    if (!ov) return;
    for (int i = 0; i < n; i++) free(ov[i]);
    free(ov);
}


static char *assemble(const FragmentArray *fa, const int *order, int n, int **ov) {
    /* Calculate exact length to avoid repeated realloc. */
    int total = (int)strlen(fa->items[order[0]]);
    for (int k = 1; k < n; k++)
        total += (int)strlen(fa->items[order[k]]) - ov[order[k - 1]][order[k]];

    char *s = malloc(total + 1);
    if (!s) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }

    strcpy(s, fa->items[order[0]]);
    for (int k = 1; k < n; k++)
        strcat(s, fa->items[order[k]] + ov[order[k - 1]][order[k]]);

    return s;
}


static void update_best(const char *candidate, bool optimal) {
    int len = (int)strlen(candidate);
    if (len < best_length) {
        free(best_solution);
        best_solution = strdup(candidate);
        best_length   = len;
        fprintf(stderr, "  New best: length %d%s — %s\n",
                len,
                optimal ? " [PROVEN OPTIMAL]" : " [greedy]",
                candidate);
    }
    if (optimal) is_optimal = true;
}

/* ═══════════════════════════════════════════════════════════════════
 * PHASE 1 — GREEDY SOLVER
 *
 * Repeatedly selects the (tail, head) pair across all current chains
 * with the maximum overlap, and merges those two chains. This is the
 * standard greedy superstring heuristic.
 *
 * Correctness: every fragment is placed in the chain exactly once, so
 * the assembled string contains every fragment.
 *
 * Optimality: NOT guaranteed. The greedy algorithm has a known worst-
 * case approximation ratio of ≤ 4 (i.e., it may produce a string up
 * to 4× longer than optimal). In practice it is usually much closer.
 *
 * Time: O(n^2) merge rounds × O(n^2) scan per round = O(n^3) overall,
 * though in practice the scan terminates as chains grow and candidates
 * shrink, making it behave like O(n^2).
 * ═══════════════════════════════════════════════════════════════════ */

static void greedy_solve(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    if (n == 0) return;
    if (n == 1) { update_best(fa->items[0], false); return; }


    int *nxt  = malloc(n * sizeof(int));
    int *prv  = malloc(n * sizeof(int));
    int *head = malloc(n * sizeof(int));
    int *tail = malloc(n * sizeof(int));
    if (!nxt || !prv || !head || !tail) {
        fprintf(stderr, "Memory allocation failed.\n"); exit(1);
    }
    for (int i = 0; i < n; i++) {
        nxt[i] = -1; prv[i] = -1; head[i] = i; tail[i] = i;
    }

    /* Perform n-1 merges to reduce to one chain. */
    for (int round = 0; round < n - 1; round++) {
        int bi = -1, bj = -1, bov = -1;

        /* Find the tail→head merge with maximum overlap. */
        for (int i = 0; i < n; i++) {
            if (nxt[i] != -1) continue;           /* i must be a tail  */
            for (int j = 0; j < n; j++) {
                if (prv[j] != -1)   continue;     /* j must be a head  */
                if (head[i] == j)   continue;     /* would create cycle */
                if (i == j)         continue;
                if (ov[i][j] > bov) { bov = ov[i][j]; bi = i; bj = j; }
            }
        }
        if (bi == -1) break;   /* no valid merge found (shouldn't happen) */

        /* Merge: chain ending at bi now continues with chain starting at bj. */
        int new_head = head[bi];
        int new_tail = tail[bj];
        nxt[bi] = bj;
        prv[bj] = bi;
        /* Propagate updated head/tail through both chains. */
        for (int cur = bj;       cur != -1; cur = nxt[cur]) head[cur] = new_head;
        for (int cur = new_head; cur != -1; cur = nxt[cur]) tail[cur] = new_tail;
    }

    /* Extract traversal order from the single remaining chain. */
    int start = -1;
    for (int i = 0; i < n; i++) if (prv[i] == -1) { start = i; break; }

    int *order = malloc(n * sizeof(int));
    if (!order) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    int cnt = 0;
    for (int cur = start; cur != -1; cur = nxt[cur])
        order[cnt++] = cur;

    /* Safety fallback: include any fragment not reached (should not occur). */
    for (int i = 0; i < n; i++) {
        bool found = false;
        for (int k = 0; k < cnt; k++) if (order[k] == i) { found = true; break; }
        if (!found) order[cnt++] = i;
    }

    char *result = assemble(fa, order, n, ov);
    update_best(result, false);

    free(result); free(order); free(nxt); free(prv); free(head); free(tail);
}

/* ═══════════════════════════════════════════════════════════════════
 * PHASE 2 — HELD-KARP EXACT DP
 *
 * Solves the Shortest Superstring Problem optimally via bitmask DP,
 * the same technique as Held-Karp for TSP.
 *
 * State:
 *   dp [mask * n + j] = maximum total overlap achievable by visiting
 *                       exactly the fragments in 'mask', ending at j.
 *   par[mask * n + j] = the fragment visited just before j on that
 *                       path (used for traceback).
 *
 * Recurrence (for each mask, each j ∈ mask, each i ∈ mask\{j}):
 *   dp[mask][j] = max over i of  dp[mask ^ (1<<j)][i] + ov[i][j]
 *
 * Base case: dp[1<<j][j] = 0  (single-fragment chains, zero overlap).
 *
 * Answer: the permutation ending at argmax_j dp[full_mask][j] gives
 * the maximum total overlap, hence the minimum superstring length.
 *
 * Maximising overlap is equivalent to minimising superstring length
 * because: |superstring| = Σ|frag_i| - total_overlap.
 *
 * Correctness: the DP evaluates all n! orderings implicitly, so the
 * solution is provably optimal.
 *
 * Flat 1-D arrays (dp[mask*n + j]) are used rather than a 2-D array
 * of pointers for better cache locality during the inner loop.
 *
 * Time:  O(2^n · n^2)
 * Space: O(2^n · n)
 * ═══════════════════════════════════════════════════════════════════ */

static void exact_solve(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    if (n == 0) return;
    if (n == 1) { update_best(fa->items[0], true); return; }

    int       full = (1 << n) - 1;
    long long sz   = (long long)(full + 1) * n;

    int *dp  = malloc(sz * sizeof(int));
    int *par = malloc(sz * sizeof(int));
    if (!dp || !par) {
        fprintf(stderr,
            "  Insufficient memory for exact DP "
            "(n=%d, would need ~%lld MB). Keeping greedy solution.\n",
            n, sz * 2 * (long long)sizeof(int) / (1024 * 1024));
        free(dp); free(par);
        return;
    }

    /* Initialise all states as unreachable. */
    for (long long k = 0; k < sz; k++) { dp[k] = -1; par[k] = -1; }
    /* Base case: each fragment alone, zero overlap accumulated. */
    for (int i = 0; i < n; i++) dp[(1 << i) * n + i] = 0;

    /* Fill DP table in order of increasing mask popcount. */
    for (int mask = 1; mask <= full && !interrupted; mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;
            int vi = dp[mask * n + i];
            if (vi < 0) continue;                /* unreachable state */
            /* Extend to each fragment j not yet in this subset. */
            for (int j = 0; j < n; j++) {
                if (mask & (1 << j)) continue;
                int nm = mask | (1 << j);
                int nv = vi + ov[i][j];
                if (nv > dp[nm * n + j]) {
                    dp[nm * n + j]  = nv;
                    par[nm * n + j] = i;
                }
            }
        }
    }

    if (!interrupted) {
        /* Find the ending fragment with maximum total overlap. */
        int best_end = 0, best_ov = -1;
        for (int i = 0; i < n; i++)
            if (dp[full * n + i] > best_ov) {
                best_ov  = dp[full * n + i];
                best_end = i;
            }

        /* Traceback to reconstruct the optimal ordering. */
        int *order = malloc(n * sizeof(int));
        if (!order) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
        int mask = full, cur = best_end;
        for (int k = n - 1; k >= 0; k--) {
            order[k] = cur;
            int prev  = par[mask * n + cur];
            mask ^= (1 << cur);
            cur   = prev;
        }
        char *result = assemble(fa, order, n, ov);
        update_best(result, true);
        free(result); free(order);
    }

    free(dp); free(par);
}


static bool verify(const char *solution, const FragmentArray *orig) {
    bool ok = true;
    printf("\nVerification against original fragments:\n");
    for (int i = 0; i < orig->count; i++) {
        bool found = (strstr(solution, orig->items[i]) != NULL);
        printf("  [%d] %-14s : %s\n",
               i, orig->items[i], found ? "FOUND" : "MISSING");
        if (!found) ok = false;
    }
    return ok;
}


static void print_fragments(const FragmentArray *fa, const char *title) {
    printf("%s (%d fragment(s)):\n", title, fa->count);
    for (int i = 0; i < fa->count; i++)
        printf("  [%d] %s\n", i, fa->items[i]);
}

static void print_overlap_matrix(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    printf("\nOverlap matrix (ov[i][j] = suffix of row i matching prefix of col j):\n");
    printf("%14s", "");
    for (int j = 0; j < n; j++) printf("%14s", fa->items[j]);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%14s", fa->items[i]);
        for (int j = 0; j < n; j++) printf("%14d", ov[i][j]);
        printf("\n");
    }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr,
            "Usage: %s [ <input_file> | - ]\n"
            "  One fragment per non-blank line.\n"
            "  Press Ctrl+C to stop and print best result found so far.\n",
            argv[0]);
        return 1;
    }


#if defined(_POSIX_VERSION)
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigaction(SIGINT, &sa, NULL);
#else
    signal(SIGINT, handle_sigint);   /* fallback for non-POSIX builds */
#endif

    /* ── 1. Read ─────────────────────────────────────────────────── */
    FragmentArray orig, work;
    fa_init(&orig); fa_init(&work);
    if (!read_fragments(argv[1], &orig)) { fa_free(&orig); return 1; }
    if (orig.count == 0) {
        fprintf(stderr, "Error: no fragments found in input.\n");
        fa_free(&orig);
        return 1;
    }
    fa_copy(&orig, &work);

    print_fragments(&orig, "Original input fragments");

    /* ── 2. Preprocess ───────────────────────────────────────────── */
    int orig_count = orig.count;
    preprocess(&work);
    printf("\n");
    print_fragments(&work, "After preprocessing (duplicates and substrings removed)");

    /* ── 3. Overlap matrix ───────────────────────────────────────── */
    int **ov = build_overlap_matrix(&work);
    print_overlap_matrix(&work, ov);

    /* ── 4. Greedy (Phase 1) ─────────────────────────────────────── */
    fprintf(stderr, "\nPhase 1: greedy solver...\n");
    greedy_solve(&work, ov);

    /* ── 5. Exact DP (Phase 2) ───────────────────────────────────── */
    if (!interrupted) {
        if (work.count > MAX_EXACT) {
            fprintf(stderr,
                "\nPhase 2: skipping exact solver (n=%d > MAX_EXACT=%d).\n"
                "  Greedy solution is the best available.\n",
                work.count, MAX_EXACT);
        } else {
            fprintf(stderr,
                "\nPhase 2: Held-Karp exact solver (n=%d, up to %d states)...\n",
                work.count, 1 << work.count);
            exact_solve(&work, ov);
        }
    } else {
        fprintf(stderr, "\nInterrupted by user. Using best solution found so far.\n");
    }

    /* ── 6. Output ───────────────────────────────────────────────── */
    if (!best_solution) {
        fprintf(stderr, "No solution produced.\n");
        free_overlap_matrix(ov, work.count);
        fa_free(&work); fa_free(&orig);
        return 1;
    }

    int sum_len = 0;
    for (int i = 0; i < work.count; i++)
        sum_len += (int)strlen(work.items[i]);

    printf("\nReconstructed string:\n  %s\n", best_solution);
    printf("\nStatistics:\n");
    printf("  Original fragment count     : %d\n",   orig_count);
    printf("  After preprocessing         : %d\n",   work.count);
    printf("  Sum of reduced lengths      : %d\n",   sum_len);
    printf("  Total overlap used          : %d\n",   sum_len - best_length);
    printf("  Final reconstructed length  : %d\n",   best_length);
    printf("  Optimality                  : %s\n",
           is_optimal    ? "PROVEN OPTIMAL"              :
           interrupted   ? "NOT PROVEN (interrupted)"    :
                           "NOT PROVEN (n > MAX_EXACT)");

    bool ok = verify(best_solution, &orig);
    printf("\nCorrectness check: %s\n", ok ? "PASS" : "FAIL");

    free_overlap_matrix(ov, work.count);
    fa_free(&work);
    fa_free(&orig);
    free(best_solution);
    return ok ? 0 : 1;
}
