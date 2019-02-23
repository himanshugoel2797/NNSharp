__kernel void activ(const int LEN,
                    const __global float* A,
#ifdef HADAMARD
                    const __global float* C,
#endif
                    __global float* B
                    ) {
    const int g_row = get_global_id(0) * WPT; // WPT = Work per thread
    
	for(int w = 0; w < WPT; w++){
		const float res = A[g_row + w];
        float activ_res = 0;

		REPLACE_THIS

#ifdef HADAMARD
        activ_res *= C[g_row + w];
#endif
        if(g_row + w < LEN)
		    B[g_row + w] = activ_res;
	}
}