__kernel void tmv_mmult(const int M, const int N,
                      const __global float* A,
                      const __global float* B,
                      const __global float* C,
                      __global float* D) {
    // Thread identifiers
    //const int col = get_local_id(0); // Local col ID (max: TS)
    const int globalCol = get_global_id(0); // Col ID of C (0..M)
    
    // Local memory to fit a tile of TS*TS elements of A and B
    float Asub[TS];
    float Bsub[TS];
 
    float acc = 0.0f;

    const int numTiles = M / TS;
    for(int t = 0; t < numTiles; t++) {
        
        for(int w = 0; w < TS; w++){
            const int row_base = t * TS + w;
            Asub[w] = A[globalCol * M + row_base];
            Bsub[w] = B[row_base];
        }

        //barrier(CLK_LOCAL_MEM_FENCE);

        for(int w = 0; w < TS; w++){
            acc += Asub[w] * Bsub[w];
        }

        //barrier(CLK_LOCAL_MEM_FENCE);
    }

    D[globalCol] = acc * C[globalCol];
}