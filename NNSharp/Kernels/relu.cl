__kernel void relu(__global float* A) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		const float a = A[globalRow + w];
		A[globalRow + w] = isgreater(a, 0) * a;
	}
}