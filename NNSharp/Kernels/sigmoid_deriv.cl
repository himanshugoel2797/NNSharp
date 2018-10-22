__kernel void sigmoid_deriv(const __global float* A, __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		const float a = 1.0f / (1.0f + exp(-A[globalRow + w]));
		B[globalRow + w] = a * (1 - a);
	}
}