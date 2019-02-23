__kernel void vc_mult(__global float* A,
					  const float B) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		A[globalRow + w] *= B;
	}
}