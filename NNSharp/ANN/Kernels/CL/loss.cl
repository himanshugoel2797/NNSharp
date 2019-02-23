__kernel void loss(const int LEN,
                    const __global float* OUT,
                    const __global float* EXPECTED,
                    __global float* B
                    ) {
        const int g_row = get_global_id(0) * WPT; // WPT = Work per thread
    
	for(int w = 0; w < WPT; w++){
		const float o = OUT[g_row + w];
		const float eo = EXPECTED[g_row + w];
                float activ_res = 0;

		REPLACE_THIS

                if(g_row + w < LEN)
                        B[g_row + w] += activ_res / LEN;
	}
}