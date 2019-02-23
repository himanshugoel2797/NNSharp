__kernel void fmop(const int LEN,
                   const float rateA,
                   const float rateB,
                   __global const float* A,
                   __global float* B
                  ) {
        const int g_row = get_global_id(0) * WPT; // WPT = Work per thread

        for(int w = 0; w < WPT; w++)
                if(g_row + w < LEN)
                        B[g_row + w] = mad(A[g_row + w], rateA, B[g_row + w] * rateB);
}