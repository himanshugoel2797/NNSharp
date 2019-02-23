__kernel void inner_prod(const int COLS, 
                         const int ROWS,
                         const __global float* A,
                         const __global float* B,
                         __global float* C) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    if(globalCol < COLS && globalRow < ROWS)
        C[globalCol * ROWS + globalRow] += A[globalRow] * B[globalCol];
}