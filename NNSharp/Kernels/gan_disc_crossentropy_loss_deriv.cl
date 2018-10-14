__kernel void gan_disc_crossentropy_loss_deriv(const __global float* A,
							 const __global float* B,
							 __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	//A = output
	//B = expectedoutput
	for(int w = 0; w < WPT; w++){
		float diff = ( A[globalRow + w] - B[globalRow + w] ) / ((A[globalRow + w] - 1) * A[globalRow + w]);
		C[globalRow + w] = -diff;
	}
}