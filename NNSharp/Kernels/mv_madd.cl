__kernel void mv_madd(const int M, const int N,
                      const __global float* A,
                      const __global float* B,
                      const __global float* C,
                      __global float* D) {
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    
    // Local memory to fit a tile of TS*TS elements of A and B
    float Asub[TS];
    float Bsub[TS];
 
    float acc = 0.0f;

    const int numTiles = N / TS;
    for(int t = 0; t < numTiles; t++) {
        
        for(int w = 0; w < TS; w++){
            const int col_base = t * TS + w;
            Asub[w] = A[col_base * M + globalRow];
            Bsub[w] = B[col_base];
        }

        //barrier(CLK_LOCAL_MEM_FENCE);

        for(int w = 0; w < TS; w++){
            acc += Asub[w] * Bsub[w];
        }

        //barrier(CLK_LOCAL_MEM_FENCE);
    }

    D[globalRow] = acc + C[globalRow];
}