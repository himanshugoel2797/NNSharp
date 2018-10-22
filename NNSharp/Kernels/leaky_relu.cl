__kernel void leaky_relu(const __global float* A, __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)

	for(int w = 0; w < WPT; w++){
		const float a = A[globalRow + w];
		B[globalRow + w] = isgreater(a, 0) * a + isless(a, 0) * 0.01f * a;
	}
}