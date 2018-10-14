__kernel void m_msub(const float rate, const int M,
					 const __global float* A,
                     __global float* B) {
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Row ID of C (0..M)
    
    B[globalRow * M + globalCol] -= A[globalRow * M + globalCol] * rate;
}