#include "atoms/affine.h"
#include "expr.h"
#include "minunit.h"
#include "utils/sparse_matrix.h"
#include "utils/tracked_alloc.h"
#include <stdlib.h>

typedef struct counting_expr
{
    expr base;
    int *forward_count;
    int *jacobian_count;
    int *hessian_count;
} counting_expr;

static void counting_forward(expr *node, const double *u)
{
    counting_expr *cnode = (counting_expr *) node;
    (*cnode->forward_count)++;
    node->value[0] = u[0];
}

static void counting_jacobian_init(expr *node)
{
    CSR_matrix *jac = new_CSR_matrix(1, 1, 1);
    jac->p[0] = 0;
    jac->p[1] = 1;
    jac->i[0] = 0;
    jac->x[0] = 1.0;
    node->jacobian = new_sparse_matrix(jac);
}

static void counting_eval_jacobian(expr *node)
{
    counting_expr *cnode = (counting_expr *) node;
    (*cnode->jacobian_count)++;
    node->jacobian->x[0] = 1.0;
}

static void counting_wsum_hess_init(expr *node)
{
    node->wsum_hess = new_sparse_matrix_alloc(1, 1, 0);
}

static void counting_eval_wsum_hess(expr *node, const double *w)
{
    counting_expr *cnode = (counting_expr *) node;
    (void) w;
    (*cnode->hessian_count)++;
}

static bool counting_is_affine(const expr *node)
{
    (void) node;
    return false;
}

static expr *new_counting_expr(int *forward_count, int *jacobian_count,
                               int *hessian_count)
{
    counting_expr *cnode = (counting_expr *) sp_calloc(1, sizeof(counting_expr));
    expr *node = &cnode->base;
    init_expr(node, 1, 1, 1, counting_forward, counting_jacobian_init,
              counting_eval_jacobian, counting_is_affine,
              counting_wsum_hess_init, counting_eval_wsum_hess, NULL);
    cnode->forward_count = forward_count;
    cnode->jacobian_count = jacobian_count;
    cnode->hessian_count = hessian_count;
    return node;
}

const char *test_shared_subexpression_runtime_cache(void)
{
    int forward_count = 0;
    int jacobian_count = 0;
    int hessian_count = 0;
    double u1[1] = {3.0};
    double u2[1] = {4.0};
    double w[1] = {1.0};

    expr *shared = new_counting_expr(&forward_count, &jacobian_count,
                                     &hessian_count);
    expr *node = new_add(shared, shared);

    expr_forward(node, u1);
    mu_assert("shared forward should be evaluated once in a DAG pass",
              forward_count == 1);
    mu_assert("shared forward value fail", node->value[0] == 6.0);

    expr_forward(node, u2);
    mu_assert("new forward pass should recompute shared node",
              forward_count == 2);
    mu_assert("second shared forward value fail", node->value[0] == 8.0);

    jacobian_init(node);
    expr_eval_jacobian(node);
    mu_assert("shared jacobian should be evaluated once in a DAG pass",
              jacobian_count == 1);
    mu_assert("shared jacobian value fail", node->jacobian->x[0] == 2.0);

    wsum_hess_init(node);
    expr_eval_wsum_hess(node, w);
    mu_assert("shared Hessian should be evaluated once for identical weights",
              hessian_count == 1);

    free_expr(node);
    return 0;
}
