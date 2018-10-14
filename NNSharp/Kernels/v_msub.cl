__kernel void v_msub(const float rate,
					 const __global const float* A,
                     const __global const float* B,
					 __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	float a_Val[WPT];
	float b_Val[WPT];

	for(int w=0; w < WPT; w++){
		a_Val[w] = A[globalRow + w];
		b_Val[w] = B[globalRow + w];
	}

	for(int w=0; w < WPT; w++){
		//C[globalRow + w] = B[globalRow + w] - A[globalRow + w] * rate;
		
		C[globalRow + w] = b_Val[w] - a_Val[w] * rate;
	}
}