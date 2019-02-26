//ROWS_A
//COLS_A
//ROWS_B
//COLS_B

//TRANSPOSE_A
//TRANSPOSE_B
//TRANSPOSE_OUT

#define M_COORD(row, col) (col * ROWS + row)

__kernel void gemm(const __global float* A,  //Matrix A
                   const __global float* B,  //Matrix B
                   const __global float* C,  //Optional-Additional term
                   __global float* D         //Output Matrix
                   ) {
    // Thread identifiers
    //const int l_row = get_local_id(0); // Local row ID (max: TS)
    //const int g_row = TS*get_group_id(0); // Row ID of C (0..M)
    const int n_row = get_global_id(0);

    //Compute the dot product of the vector
    float acc = 0.0f;
#ifdef TRANSPOSE_A
    for(int t = 0; t < ROWS; t++){
        acc += A[M_COORD(t, n_row)] * B[t];
    }
#else
    for(int t = 0; t < COLS; t++){
        acc += A[M_COORD(n_row, t)] * B[t];
    }
#endif

#ifdef TRANSPOSE_A
    if(n_row < COLS)
#else
    if(n_row < ROWS)
#endif
    {
#ifdef S_OP_ADD
        D[n_row] = acc + C[n_row];
#else
        D[n_row] = acc;
#endif
    }
}