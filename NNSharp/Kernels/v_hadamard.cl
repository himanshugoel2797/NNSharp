__kernel void v_hadamard(const __global float* A,
                         const __global float* B,
                         __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		C[globalRow + w] = A[globalRow + w] * B[globalRow + w];
	}
}