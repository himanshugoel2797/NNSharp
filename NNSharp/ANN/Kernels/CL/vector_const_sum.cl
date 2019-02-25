//O_LEN

__kernel void vector_const_sum(__global float* OUT,
                         const __global float* IN,
                         const int IN_off
                         ){
        const int g_row = get_global_id(0) * WPT; // WPT = Work per thread

        const float i = IN[IN_off];

        for(int w = 0; w < WPT; w++)
                if(g_row + w < O_LEN)
                        OUT[g_row + w] += i;
}