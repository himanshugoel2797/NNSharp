__kernel void v_act(const __global float* A,
                     __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		const float res = A[globalRow + w];
		REPLACE_THIS

		B[globalRow + w] = activ_res;
	}
}