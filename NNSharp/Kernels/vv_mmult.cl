__kernel void vv_mmult(const int M, const int N,
                       const __global float* A,
                       const __global float* B,
                       __global float* C) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    C[globalCol * M + globalRow] += A[globalRow] * B[globalCol];
}