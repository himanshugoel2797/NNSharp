#define TS 16

__kernel void mv_madd_0(const int M, const int N,
                      const __global float* A,
                      const __global float* B,
                      const __global float* C,
                      __global float* D) {
    // Thread identifiers
    const int globalRow = get_group_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1);

    D[globalCol * M + globalRow] = A[globalCol * M + globalRow] * B[globalCol * M + globalRow];
}