__kernel void conv_bkwd_err_small(const int M, /*inputSz Eg: 1024*/
                            const int N, /*filterSz Eg: 1022*/
                            const int K, /*outputSz Eg: 3*/
                            const int S, /*strideLen Eg: 1*/
                            const int P, /*padding Eg: 0*/
                            const int OX, /*origin_base_x Eg: 1022/2*/
                            const int OY, /*origin_base_y Eg: 1022/2*/
                            const int OZ, /*i/filter index Eg: 0*/
                            const int OW, /*j/input depth Eg: 0*/
                            const __global float* kern,  /*convolution kernel Eg: 1022x1022 Output*/
                            const __global float* i, /*input*/
                            __global float* o) {
    const int globalRow = get_global_id(0); // Row ID of C (0..M)

    __local float kern_s[400];
    for(int q = 0; q < N * N; q++)
        kern_s[q] = kern[N*N*OW + q];

    barrier(CLK_LOCAL_MEM_FENCE);

    //Output position    
    const int row = (globalRow % K) * S;
    const int col = (globalRow / K) * S;

    float acc = 0.0f;

    for(int q = 0; q < N * N; q++){
        const int l_row = q % N;
        const int l_col = q / N;

        const int i_row = row + l_row;
        const int i_col = col + l_col;

        if(i_row >= P && i_col >= P && i_row < M + P && i_col < M + P){
            acc += i[M * M * OZ + (i_col - P) * M + (i_row - P)] * kern_s[(l_col * N + l_row)];
        }
        //acc += i[0] * kern[l_col * N + l_row];
    }

    o[col * K + row] += acc;
}