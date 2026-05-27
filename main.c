#include "base.h"
#include "arena.h"

#include "arena.c"

typedef struct {
    u32 rows, cols;
    f32 data;
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
void mat_clear(matrix* mat);
void copy_mat(matirx* dst, matrix* src);
void mat_fill(matrix* mat, f32 x);
void mat_scale(matrix* mat, f32 scale);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);


int main() {

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    

    arena_destroy(perm_arena);

    return 0;
}

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);

void mat_clear(matrix* mat);

void copy_mat(matirx* dst, matrix* src);

void mat_fill(matrix* mat, f32 x);

void mat_scale(matrix* mat, f32 scale);

b32 mat_add(matrix* out, const matrix* a, const matrix* b);

b32 mat_sub(matrix* out, const matrix* a, const matrix* b);

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);

b32 mat_relu(matrix* out, const matrix* in);

b32 mat_softmax(matrix* out, const matrix* in);

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);

b32 mat_relu_add_grad(matrix* out, const matrix* in);

b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);

b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);
