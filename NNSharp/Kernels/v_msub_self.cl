__kernel void v_msub_self(const float rate,
					 const __global const float* A,
                     __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	float a_Val[WPT];

	for(int w=0; w < WPT; w++){
		a_Val[w] = A[globalRow + w];
	}

	for(int w=0; w < WPT; w++){
		//C[globalRow + w] = B[globalRow + w] - A[globalRow + w] * rate;
		
		B[globalRow + w] -= a_Val[w] * rate;
	}
}