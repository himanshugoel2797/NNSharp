__kernel void gan_gen_crossentropy_loss_deriv(const __global float* A,
							 const __global float* B,
							 __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	//A = output from generative model
	//B = output from real model

	for(int w = 0; w < WPT; w++){
		float diff = (-0.5f * 1.0f / A[globalRow + w]) - 0.5f * 1.0f / (1.0f - B[globalRow + w]);
		C[globalRow + w] = (diff);
	}
}