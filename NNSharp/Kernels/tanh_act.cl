__kernel void tanh_act(__global float* A) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		A[globalRow + w] = 1.7159f * tanh(0.6667f * A[globalRow + w]);
	}
}