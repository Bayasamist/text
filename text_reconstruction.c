#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    char **items;
    int count;
    int capacity;
} FragmentArray;

typedef struct {
    int *order;
    int count;
    int total_overlap;
} DPSolution;

static void init_fragment_array(FragmentArray *fa) {
    fa->items = NULL;
    fa->count = 0;
    fa->capacity = 0;
}

static void free_fragment_array(FragmentArray *fa) {
    if (fa->items != NULL) {
        for (int i = 0; i < fa->count; i++) {
            free(fa->items[i]);
        }
        free(fa->items);
    }
    fa->items = NULL;
    fa->count = 0;
    fa->capacity = 0;
}

static void append_fragment(FragmentArray *fa, const char *text) {
    if (fa->count == fa->capacity) {
        int new_capacity = (fa->capacity == 0) ? 8 : fa->capacity * 2;
        char **new_items = realloc(fa->items, new_capacity * sizeof(char *));
        if (new_items == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
        fa->items = new_items;
        fa->capacity = new_capacity;
    }

    char *copy = strdup(text);
    if (copy == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    fa->items[fa->count++] = copy;
}

static void copy_fragment_array(const FragmentArray *src, FragmentArray *dst) {
    init_fragment_array(dst);
    for (int i = 0; i < src->count; i++) {
        append_fragment(dst, src->items[i]);
    }
}

static void trim_newline(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[len - 1] = '\0';
        len--;
    }
}

static bool read_fragments(const char *filename, FragmentArray *fa) {
    FILE *fp = NULL;

    if (strcmp(filename, "-") == 0) {
        fp = stdin;
    } else {
        fp = fopen(filename, "r");
    }

    if (fp == NULL) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return false;
    }

    char *line = NULL;
    size_t cap = 0;

    while (getline(&line, &cap, fp) != -1) {
        trim_newline(line);
        if (line[0] == '\0') {
            continue;
        }
        append_fragment(fa, line);
    }

    free(line);

    if (fp != stdin) {
        fclose(fp);
    }

    return true;
}

static void print_fragments(const FragmentArray *fa, const char *title) {
    printf("%s (%d fragment(s)):\n", title, fa->count);
    for (int i = 0; i < fa->count; i++) {
        printf("[%d] %s\n", i, fa->items[i]);
    }
}

static int total_fragment_length(const FragmentArray *fa) {
    int total = 0;
    for (int i = 0; i < fa->count; i++) {
        total += (int)strlen(fa->items[i]);
    }
    return total;
}

static bool is_substring(const char *small, const char *big) {
    return strstr(big, small) != NULL;
}

static void remove_duplicates(FragmentArray *fa) {
    for (int i = 0; i < fa->count; i++) {
        int j = i + 1;
        while (j < fa->count) {
            if (strcmp(fa->items[i], fa->items[j]) == 0) {
                free(fa->items[j]);
                for (int k = j; k < fa->count - 1; k++) {
                    fa->items[k] = fa->items[k + 1];
                }
                fa->count--;
            } else {
                j++;
            }
        }
    }
}

static void remove_contained_fragments(FragmentArray *fa) {
    int i = 0;
    while (i < fa->count) {
        bool contained = false;

        for (int j = 0; j < fa->count; j++) {
            if (i == j) {
                continue;
            }
            if (is_substring(fa->items[i], fa->items[j])) {
                contained = true;
                break;
            }
        }

        if (contained) {
            free(fa->items[i]);
            for (int k = i; k < fa->count - 1; k++) {
                fa->items[k] = fa->items[k + 1];
            }
            fa->count--;
        } else {
            i++;
        }
    }
}

static void preprocess_fragments(FragmentArray *fa) {
    remove_duplicates(fa);
    remove_contained_fragments(fa);
}

static int overlap_length(const char *left, const char *right) {
    int left_len = (int)strlen(left);
    int right_len = (int)strlen(right);
    int max_possible = left_len < right_len ? left_len : right_len;

    for (int k = max_possible; k >= 1; k--) {
        bool match = true;
        for (int i = 0; i < k; i++) {
            if (left[left_len - k + i] != right[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return k;
        }
    }

    return 0;
}

static int **build_overlap_matrix(const FragmentArray *fa) {
    int n = fa->count;

    int **overlap = malloc(n * sizeof(int *));
    if (overlap == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        overlap[i] = malloc(n * sizeof(int));
        if (overlap[i] == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            overlap[i][j] = (i == j) ? 0 : overlap_length(fa->items[i], fa->items[j]);
        }
    }

    return overlap;
}

static void free_overlap_matrix(int **overlap, int n) {
    if (overlap == NULL) {
        return;
    }
    for (int i = 0; i < n; i++) {
        free(overlap[i]);
    }
    free(overlap);
}

static void print_overlap_matrix(const FragmentArray *fa, int **overlap) {
    int n = fa->count;

    printf("\nOverlap matrix:\n");
    printf("%8s", "");
    for (int j = 0; j < n; j++) {
        printf("%8s", fa->items[j]);
    }
    printf("\n");

    for (int i = 0; i < n; i++) {
        printf("%8s", fa->items[i]);
        for (int j = 0; j < n; j++) {
            printf("%8d", overlap[i][j]);
        }
        printf("\n");
    }
}

static DPSolution solve_exact_dp(const FragmentArray *fa, int **overlap) {
    int n = fa->count;
    int total_masks = 1 << n;

    int **dp = malloc(total_masks * sizeof(int *));
    int **parent = malloc(total_masks * sizeof(int *));
    if (dp == NULL || parent == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (int mask = 0; mask < total_masks; mask++) {
        dp[mask] = malloc(n * sizeof(int));
        parent[mask] = malloc(n * sizeof(int));
        if (dp[mask] == NULL || parent[mask] == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }

        for (int j = 0; j < n; j++) {
            dp[mask][j] = -1;
            parent[mask][j] = -1;
        }
    }

    for (int j = 0; j < n; j++) {
        dp[1 << j][j] = 0;
    }

    for (int mask = 1; mask < total_masks; mask++) {
        for (int j = 0; j < n; j++) {
            if (!(mask & (1 << j))) {
                continue;
            }

            int prev_mask = mask ^ (1 << j);
            if (prev_mask == 0) {
                continue;
            }

            for (int i = 0; i < n; i++) {
                if (!(prev_mask & (1 << i))) {
                    continue;
                }
                if (dp[prev_mask][i] == -1) {
                    continue;
                }

                int candidate = dp[prev_mask][i] + overlap[i][j];
                if (candidate > dp[mask][j]) {
                    dp[mask][j] = candidate;
                    parent[mask][j] = i;
                }
            }
        }
    }

    int full_mask = total_masks - 1;
    int best_end = 0;
    int best_overlap = -1;

    for (int j = 0; j < n; j++) {
        if (dp[full_mask][j] > best_overlap) {
            best_overlap = dp[full_mask][j];
            best_end = j;
        }
    }

    int *order = malloc(n * sizeof(int));
    if (order == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int mask = full_mask;
    int current = best_end;

    for (int pos = n - 1; pos >= 0; pos--) {
        order[pos] = current;
        int prev = parent[mask][current];
        mask ^= (1 << current);
        current = prev;
    }

    for (int m = 0; m < total_masks; m++) {
        free(dp[m]);
        free(parent[m]);
    }
    free(dp);
    free(parent);

    DPSolution sol;
    sol.order = order;
    sol.count = n;
    sol.total_overlap = best_overlap;
    return sol;
}

static void print_order(const FragmentArray *fa, const DPSolution *sol) {
    printf("\nOptimal order of fragments:\n");
    for (int i = 0; i < sol->count; i++) {
        int idx = sol->order[i];
        printf("%s", fa->items[idx]);
        if (i + 1 < sol->count) {
            printf(" -> ");
        }
    }
    printf("\n");
}

static char *build_superstring(const FragmentArray *fa, const DPSolution *sol, int **overlap) {
    int total_length = 0;

    for (int i = 0; i < sol->count; i++) {
        total_length += (int)strlen(fa->items[sol->order[i]]);
    }
    total_length -= sol->total_overlap;

    char *result = malloc((size_t)total_length + 1);
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    result[0] = '\0';

    int first = sol->order[0];
    strcpy(result, fa->items[first]);

    for (int pos = 1; pos < sol->count; pos++) {
        int prev = sol->order[pos - 1];
        int curr = sol->order[pos];
        int ov = overlap[prev][curr];
        strcat(result, fa->items[curr] + ov);
    }

    return result;
}

static bool verify_solution(const char *solution, const FragmentArray *original_fragments) {
    bool all_found = true;

    printf("\nVerification against original fragments:\n");
    for (int i = 0; i < original_fragments->count; i++) {
        const char *frag = original_fragments->items[i];
        bool found = is_substring(frag, solution);
        printf("[%d] %-10s : %s\n", i, frag, found ? "FOUND" : "MISSING");
        if (!found) {
            all_found = false;
        }
    }

    return all_found;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file | ->\n", argv[0]);
        return 1;
    }

    FragmentArray original_fragments;
    FragmentArray fragments;
    init_fragment_array(&original_fragments);
    init_fragment_array(&fragments);

    if (!read_fragments(argv[1], &original_fragments)) {
        free_fragment_array(&original_fragments);
        free_fragment_array(&fragments);
        return 1;
    }

    if (original_fragments.count == 0) {
        fprintf(stderr, "Error: no fragments found.\n");
        free_fragment_array(&original_fragments);
        free_fragment_array(&fragments);
        return 1;
    }

    copy_fragment_array(&original_fragments, &fragments);

    print_fragments(&original_fragments, "Original input fragments");

    int original_count = original_fragments.count;
    preprocess_fragments(&fragments);
    int reduced_count = fragments.count;
    int sum_lengths = total_fragment_length(&fragments);

    print_fragments(&fragments, "After preprocessing");

    int **overlap = build_overlap_matrix(&fragments);
    print_overlap_matrix(&fragments, overlap);

    DPSolution sol = solve_exact_dp(&fragments, overlap);
    print_order(&fragments, &sol);

    char *answer = build_superstring(&fragments, &sol, overlap);
    size_t final_length = strlen(answer);

    printf("\nReconstructed string:\n%s\n", answer);

    printf("\nStatistics:\n");
    printf("Original fragment count      : %d\n", original_count);
    printf("After preprocessing count    : %d\n", reduced_count);
    printf("Sum of reduced lengths       : %d\n", sum_lengths);
    printf("Total overlap used           : %d\n", sol.total_overlap);
    printf("Final reconstructed length   : %zu\n", final_length);
    printf("Length formula check         : %d - %d = %zu\n",
           sum_lengths, sol.total_overlap, final_length);

    bool ok = verify_solution(answer, &original_fragments);
    printf("\nOverall correctness check: %s\n", ok ? "PASS" : "FAIL");

    free(answer);
    free(sol.order);
    free_overlap_matrix(overlap, fragments.count);
    free_fragment_array(&fragments);
    free_fragment_array(&original_fragments);

    return ok ? 0 : 1;
}