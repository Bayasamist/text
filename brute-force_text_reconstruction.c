/*
 * IFN664 Advanced Algorithms — Group Project - Group 7
 * Minimum-Size Text Reconstruction from Overlapping Fragments
 *
 * Compile:  gcc -O2 -std=gnu11 -Wall -Wextra -o text_reconstruction text_reconstruction.c
 * Usage:    ./text_reconstruction <input_file>
 *           ./text_reconstruction -          (read from stdin)
 *           Press Ctrl+C to stop early and keep the best solution found so far.
 *
 * Algorithm overview:
 *   1. Read fragments from input file.
 *   2. Preprocess: remove duplicates and fragments contained in others.
 *      (Safe: any superstring covering the larger fragment already covers the smaller.)
 *   3. Build pairwise overlap matrix: ov[i][j] = length of longest suffix of fragment i
 *      that equals a prefix of fragment j.
 *   4. Greedy solver: repeatedly merge the pair with greatest overlap.
 *      Runs in O(n^2) after the overlap matrix is built. Outputs a solution immediately.
 *   5. Exact solver: Held-Karp dynamic programming over subsets.
 *      Finds the provably optimal solution. O(2^n * n^2) time, O(2^n * n) space.
 *      Skipped automatically if n > MAX_EXACT (not enough memory).
 *   6. SIGINT handler: Ctrl+C prints the best solution found so far and exits cleanly.
 *   7. Verification: confirms every original fragment appears in the final output.
 *
 * Complexity summary:
 *   Preprocessing      : O(n^2 * L^2)   n = fragments after pruning, L = max length
 *   Overlap matrix     : O(n^2 * L^2)
 *   Greedy             : O(n^2)
 *   Held-Karp exact DP : O(2^n * n^2) time,  O(2^n * n) space
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
    char **items;
    int    count;
    int    capacity;
} FragmentArray;


static volatile sig_atomic_t interrupted  = 0;
static char                 *best_solution = NULL;   /* best so far      */
static int                   best_length   = INT_MAX;
static bool                  is_optimal    = false;

static void handle_sigint(int sig) {
    (void)sig;
    interrupted = 1;
}



static void fa_init(FragmentArray *fa) {
    fa->items = NULL; fa->count = 0; fa->capacity = 0;
}

static void fa_free(FragmentArray *fa) {
    if (fa->items) {
        for (int i = 0; i < fa->count; i++) free(fa->items[i]);
        free(fa->items);
    }
    fa_init(fa);
}

static void fa_append(FragmentArray *fa, const char *text) {
    if (fa->count == fa->capacity) {
        int nc = (fa->capacity == 0) ? 8 : fa->capacity * 2;
        char **ni = realloc(fa->items, nc * sizeof(char *));
        if (!ni) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
        fa->items = ni; fa->capacity = nc;
    }
    char *copy = strdup(text);
    if (!copy) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    fa->items[fa->count++] = copy;
}

static void fa_copy(const FragmentArray *src, FragmentArray *dst) {
    fa_init(dst);
    for (int i = 0; i < src->count; i++) fa_append(dst, src->items[i]);
}


static void trim_newline(char *s) {
    size_t l = strlen(s);
    while (l > 0 && (s[l-1] == '\n' || s[l-1] == '\r')) s[--l] = '\0';
}

/* Reads one fragment per non-blank line into fa. */
static bool read_fragments(const char *filename, FragmentArray *fa) {
    FILE *fp = (strcmp(filename, "-") == 0) ? stdin : fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open '%s'\n", filename); return false; }
    char *line = NULL; size_t cap = 0;
    while (getline(&line, &cap, fp) != -1) {
        trim_newline(line);
        if (line[0] != '\0') fa_append(fa, line);
    }
    free(line);
    if (fp != stdin) fclose(fp);
    return true;
}



static void remove_duplicates(FragmentArray *fa) {
    for (int i = 0; i < fa->count; i++) {
        int j = i + 1;
        while (j < fa->count) {
            if (strcmp(fa->items[i], fa->items[j]) == 0) {
                free(fa->items[j]);
                for (int k = j; k < fa->count - 1; k++) fa->items[k] = fa->items[k+1];
                fa->count--;
            } else j++;
        }
    }
}

static void remove_contained(FragmentArray *fa) {
    int i = 0;
    while (i < fa->count) {
        bool contained = false;
        for (int j = 0; j < fa->count && !contained; j++) {
            if (i != j && strstr(fa->items[j], fa->items[i]) != NULL)
                contained = true;
        }
        if (contained) {
            free(fa->items[i]);
            for (int k = i; k < fa->count - 1; k++) fa->items[k] = fa->items[k+1];
            fa->count--;
        } else i++;
    }
}

static void preprocess(FragmentArray *fa) {
    remove_duplicates(fa);
    remove_contained(fa);
}



static int compute_overlap(const char *left, const char *right) {
    int ll = (int)strlen(left), rl = (int)strlen(right);
    int limit = ll < rl ? ll : rl;
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
    for (int i = 0; i < n; i++) {
        ov[i] = malloc((size_t)n * sizeof(int));
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
    int total = (int)strlen(fa->items[order[0]]);
    for (int k = 1; k < n; k++)
        total += (int)strlen(fa->items[order[k]]) - ov[order[k-1]][order[k]];
    char *s = malloc(total + 1);
    if (!s) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    strcpy(s, fa->items[order[0]]);
    for (int k = 1; k < n; k++)
        strcat(s, fa->items[order[k]] + ov[order[k-1]][order[k]]);
    return s;
}



static void update_best(const char *candidate, bool optimal) {
    int len = (int)strlen(candidate);
    if (len < best_length) {
        free(best_solution);
        best_solution = strdup(candidate);
        best_length   = len;
        fprintf(stderr, "  New best solution (length %d)%s: %s\n",
                len, optimal ? " [OPTIMAL]" : " [greedy]", candidate);
    }
    if (optimal) is_optimal = true;
}



static void greedy_solve(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    if (n == 0) return;
    if (n == 1) { update_best(fa->items[0], false); return; }

 
    int *nxt = malloc((unsigned)n * sizeof(int));
    int *prv = malloc((unsigned)n * sizeof(int));
    /* head[i] = head of the chain containing i
       tail[i] = tail of the chain containing i */
    int *head = malloc((unsigned)n * sizeof(int));
    int *tail = malloc((unsigned)n * sizeof(int));
    for (int i = 0; i < n; i++) { nxt[i]=-1; prv[i]=-1; head[i]=i; tail[i]=i; }

    for (int round = 0; round < n - 1; round++) {
        /* find best (tail → head) merge */
        int bi = -1, bj = -1, bov = -1;
        for (int i = 0; i < n; i++) {
            if (nxt[i] != -1) continue;   /* i must be a tail */
            for (int j = 0; j < n; j++) {
                if (prv[j] != -1) continue;           /* j must be a head */
                if (head[i] == j)  continue;          /* would create cycle */
                if (i == j)        continue;
                if (ov[i][j] > bov) { bov=ov[i][j]; bi=i; bj=j; }
            }
        }
        if (bi == -1) break;

        /* merge: chain ending at bi now continues with chain starting at bj */
        int new_head = head[bi];
        int new_tail = tail[bj];
        nxt[bi] = bj;
        prv[bj] = bi;
        /* update head/tail pointers for all nodes in the merged chain */
        for (int cur = bj; cur != -1; cur = nxt[cur]) head[cur] = new_head;
        for (int cur = new_head; cur != -1; cur = nxt[cur]) tail[cur] = new_tail;
    }

    /* extract ordering */
    int start = -1;
    for (int i = 0; i < n; i++) if (prv[i] == -1) { start = i; break; }
    int *order = malloc((unsigned)n * sizeof(int));
    int  cnt   = 0;
    for (int cur = start; cur != -1; cur = nxt[cur]) order[cnt++] = cur;
    /* safety: add any fragments not reached (shouldn't happen) */
    for (int i = 0; i < n; i++) {
        bool found = false;
        for (int k = 0; k < cnt; k++) if (order[k] == i) { found = true; break; }
        if (!found) order[cnt++] = i;
    }

    char *result = assemble(fa, order, n, ov);
    update_best(result, false);
    free(result); free(order); free(nxt); free(prv); free(head); free(tail);
}



static void exact_solve(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    if (n == 0) return;
    if (n == 1) { update_best(fa->items[0], true); return; }

    int full = (1 << n) - 1;
    long long sz = (long long)(full + 1) * n;

    int *dp  = malloc(sz * sizeof(int));
    int *par = malloc(sz * sizeof(int));
    if (!dp || !par) {
        fprintf(stderr,
            "  Not enough memory for exact DP (n=%d, need ~%lld MB).\n"
            "  Keeping greedy solution.\n",
            n, sz * 2 * sizeof(int) / (1024*1024));
        free(dp); free(par); return;
    }

    /* initialise */
    for (long long k = 0; k < sz; k++) { dp[k] = -1; par[k] = -1; }
    for (int i = 0; i < n; i++) dp[(1<<i)*n + i] = 0;

    /* fill */
    for (int mask = 1; mask <= full && !interrupted; mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1<<i))) continue;
            int vi = dp[mask*n + i];
            if (vi < 0) continue;
            for (int j = 0; j < n; j++) {
                if (mask & (1<<j)) continue;
                int nm = mask | (1<<j);
                int nv = vi + ov[i][j];
                if (nv > dp[nm*n + j]) {
                    dp[nm*n + j]  = nv;
                    par[nm*n + j] = i;
                }
            }
        }
    }

    if (!interrupted) {
        /* find best ending fragment */
        int best_end = 0, best_ov = -1;
        for (int i = 0; i < n; i++)
            if (dp[full*n + i] > best_ov) { best_ov = dp[full*n+i]; best_end = i; }

        /* reconstruct path */
        int *order = malloc((size_t)n * sizeof(int));
        int  mask  = full, cur = best_end;
        for (int k = n-1; k >= 0; k--) {
            order[k] = cur;
            int prev = par[mask*n + cur];
            mask ^= (1<<cur);
            cur = prev;
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
        printf("  [%d] %-12s : %s\n", i, orig->items[i], found ? "FOUND" : "MISSING");
        if (!found) ok = false;
    }
    return ok;
}



static void print_overlap_matrix(const FragmentArray *fa, int **ov) {
    int n = fa->count;
    printf("\nOverlap matrix:\n");
    printf("%12s", "");
    for (int j = 0; j < n; j++) printf("%12s", fa->items[j]);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%12s", fa->items[i]);
        for (int j = 0; j < n; j++) printf("%12d", ov[i][j]);
        printf("\n");
    }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr,
            "Usage: %s [ <input_file> | - ]\n"
            "  One fragment per line. Press Ctrl+C to stop and keep best result.\n",
            argv[0]);
        return 1;
    }

    /* Install SIGINT handler for graceful interruption (Ctrl+C).
     * Guarded by _POSIX_VERSION so IntelliSense on Windows (where
     * struct sigaction is undefined) sees the portable signal() fallback
     * instead. On Linux with gcc the POSIX branch is always taken. */
#if defined(_POSIX_VERSION)
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigaction(SIGINT, &sa, NULL);
#else
    signal(SIGINT, handle_sigint);   
#endif

    // 1. read 
    FragmentArray orig, work;
    fa_init(&orig); fa_init(&work);
    if (!read_fragments(argv[1], &orig)) { fa_free(&orig); return 1; }
    if (orig.count == 0) { fprintf(stderr, "No fragments found.\n"); return 1; }
    fa_copy(&orig, &work);

    printf("Original input fragments (%d):\n", orig.count);
    for (int i = 0; i < orig.count; i++) printf("  [%d] %s\n", i, orig.items[i]);

    //2. preprocess
    preprocess(&work);
    printf("\nAfter preprocessing (%d fragment(s) remain):\n", work.count);
    for (int i = 0; i < work.count; i++) printf("  [%d] %s\n", i, work.items[i]);

    //3. overlap matrix 
    int **ov = build_overlap_matrix(&work);
    print_overlap_matrix(&work, ov);

    //4. greedy (fast approximate solution)
    fprintf(stderr, "\nPhase 1: greedy solver...\n");
    greedy_solve(&work, ov);

    //5. exact DP
    if (!interrupted) {
        if (work.count > MAX_EXACT) {
            fprintf(stderr,
                "\nPhase 2: skipping exact solver (n=%d > %d).\n"
                "  Greedy solution is the best available.\n",
                work.count, MAX_EXACT);
        } else {
            fprintf(stderr, "\nPhase 2: exact Held-Karp solver (n=%d)...\n", work.count);
            exact_solve(&work, ov);
        }
    } else {
        fprintf(stderr, "\nInterrupted. Using best solution found so far.\n");
    }

    //6. output
    if (best_solution) {
        int sum_len = 0;
        for (int i = 0; i < work.count; i++) sum_len += (int)strlen(work.items[i]);

        printf("\nShortest reconstructed string:\n%s\n", best_solution);
        printf("\nStatistics:\n");
        printf("  Original fragment count     : %d\n",   orig.count);
        printf("  After preprocessing         : %d\n",   work.count);
        printf("  Sum of reduced lengths      : %d\n",   sum_len);
        printf("  Total overlap used          : %d\n",   sum_len - best_length);
        printf("  Final reconstructed length  : %d\n",   best_length);
        printf("  Optimality                  : %s\n",
               is_optimal ? "PROVEN OPTIMAL" :
               interrupted ? "NOT PROVEN (interrupted)" : "NOT PROVEN (n too large)");

        bool ok = verify(best_solution, &orig);
        printf("\nCorrectness check: %s\n", ok ? "PASS" : "FAIL");

        free_overlap_matrix(ov, work.count);
        fa_free(&work); fa_free(&orig);
        free(best_solution);
        return ok ? 0 : 1;
    }

    fprintf(stderr, "No solution produced.\n");
    free_overlap_matrix(ov, work.count);
    fa_free(&work); fa_free(&orig);
    return 1;
}