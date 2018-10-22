__kernel void v_hadamard_act(const __global float* A,
                         const __global float* B,
                         __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		const float res = B[globalRow + w];
		REPLACE_THIS

		C[globalRow + w] = A[globalRow + w] * activ_res;
	}
}