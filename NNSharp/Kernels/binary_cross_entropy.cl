__kernel void binary_cross_entropy(const int len, 
								   const __global float* A,
							       const __global float* B,
							       __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	//A = expectedOutput
	//B = actualOutput

	for(int w = 0; w < WPT; w++){
		float z = B[globalRow + w];
		float y = A[globalRow + w];

		const float eps = 1e-12;

		float diff = (A[globalRow + w] - B[globalRow + w]);
		C[globalRow + w] += - (z * log(y + eps) + (1-z) * log(1 - y + eps)) / len;
	}
}