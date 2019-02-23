__kernel void conv_fwd(const int M, /*inputSz Eg: 1024*/
                       const int N, /*filterSz Eg: 3*/
                       const int K, /*outputSz Eg: 1022*/
                       const int S, /*strideLen Eg: 1*/
                       const int P, /*padding Eg: 0*/
                       const int OX, /*origin_base_x Eg: 1*/
                       const int OY, /*origin_base_y Eg: 1*/
                       const int OZ, /*i/filter index Eg: 0*/
                       const int OW, /*j/input depth Eg: 0*/
                       const __global float* kern,  /*convolution kernel Eg: 3x3 Gaussian*/
                       const __global float* i, /*input*/
                       __global float* o) {
    const int globalRow = get_global_id(0); // Row ID of C (0..M)

    //Output position    
    const int row = (globalRow % K) * S;
    const int col = (globalRow / K) * S;

    float acc = 0.0f;

    for(int q = 0; q < N * N; q++){
        const int l_row = q % N;
        const int l_col = q / N;

        const int i_row = row + l_row;
        const int i_col = col + l_col;

        if(i_row >= P && i_col >= P && i_row <= M + P && i_col <= M + P){
            acc += i[M * M * OW + (i_col - P) * M + (i_row - P)] * kern[l_col * N + l_row];
        }
    }

    o[K * K * OZ + col * K + row] += acc;
}