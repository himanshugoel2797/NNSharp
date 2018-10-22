__kernel void tanh_act(const __global float* A, __global float* B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		B[globalRow + w] = 1.7159f * tanh(0.6667f * A[globalRow + w]);
	}
}