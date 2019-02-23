__kernel void uniform_init(const int LEN,
                    const __global float* SEED,
                    const float BIAS,
                    __global float* OUTPUT
                    ) {
        const int g_row = get_global_id(0) * WPT; // WPT = Work per thread
    
	for(int w = 0; w < WPT; w++){
		
                if(g_row + w < LEN)
                        OUTPUT[g_row + w] = activ_res;
	}
}