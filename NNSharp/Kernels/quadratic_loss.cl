﻿__kernel void quadratic_loss(const __global float* A,
							 const __global float* B,
							 __global float* C) {
    const int globalRow = get_global_id(0) * WPT; // Row ID of C (0..M)
    
	for(int w = 0; w < WPT; w++){
		float diff = (A[globalRow + w] - B[globalRow + w]);
		C[globalRow + w] = 0.5f * diff * diff;
	}
}