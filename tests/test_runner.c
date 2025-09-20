#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/neat.h"

/* Forward declarations of test functions */
void test_activation_functions();
void test_node_creation();
void test_connection_creation();
void test_genome_operations();
void test_mutations();
void test_crossover();
void test_speciation();
void test_xor_problem();
void test_performance();

/* Test statistics */
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
} TestStats;

static TestStats stats = {0};

#define TEST(cond, message) \
    do { \
        stats.total_tests++; \
        if (cond) { \
            stats.passed_tests++; \
            printf("\033[0;32mPASS\033[0m: %s\n", message); \
        } else { \
            stats.failed_tests++; \
            printf("\033[0;31mFAIL\033[0m: %s (File: %s, Line: %d)\n", message, __FILE__, __LINE__); \
        } \
    } while(0)

#define TEST_EQUAL(a, b, message) TEST((a) == (b), message)
#define TEST_NOT_EQUAL(a, b, message) TEST((a) != (b), message)
#define TEST_TRUE(cond, message) TEST((cond), message)
#define TEST_FALSE(cond, message) TEST(!(cond), message)

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void print_test_header(const char* test_name) {
    printf("\n\033[1;36m===== %s =====\033[0m\n", test_name);
}

void print_test_footer() {
    printf("\nTest Summary:");
    printf("\n  Total:  %d", stats.total_tests);
    printf("\n  \033[0;32mPassed: %d\033[0m", stats.passed_tests);
    if (stats.failed_tests > 0) {
        printf("\n  \033[0;31mFailed: %d\033[0m", stats.failed_tests);
    } else {
        printf("\n  Failed: %d", stats.failed_tests);
    }
    printf("\n\n");
}

int main() {
    double start_time = get_time();
    
    printf("\n\033[1;35m===== NEAT Test Suite =====\033[0m\n");
    
    /* Run individual test suites */
    test_activation_functions();
    test_node_creation();
    test_connection_creation();
    test_genome_operations();
    test_mutations();
    test_crossover();
    test_speciation();
    test_xor_problem();
    test_performance();
    
    double end_time = get_time();
    
    print_test_footer();
    printf("Total test execution time: %.3f seconds\n", end_time - start_time);
    
    return stats.failed_tests > 0 ? 1 : 0;
}
