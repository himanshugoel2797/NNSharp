__kernel void leaky_relu_deriv(const __global float* A, __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		B[globalRow + w] = isgreater(A[globalRow + w], 0) + isless(A[globalRow + w], 0) * 0.01f;
	}
}