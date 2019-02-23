//IN_D
//KERN_D
//OUT_D
//STRIDE

__kernel void error_maxpool(const int IN_OFF,
                            const int OUT_OFF,
                            const __global float* i,
                            const __global float* cache,
                            __global float* o) {

    const int globalRow = get_global_id(0); //row
    const int globalCol = get_global_id(1); //col

    //Output position    
    const int row = globalRow;
    const int col = globalCol;

    const float cur_val = i[OUT_OFF + col * OUT_D + row];

    for (int n0 = 0; n0 < KERN_D; n0++)
        for (int n1 = 0; n1 < KERN_D; n1++)
        {
            int i_x = col * STRIDE + (n0 - KERN_D / 2) + KERN_D / 2;
            int i_y = row * STRIDE + (n1 - KERN_D / 2) + KERN_D / 2;

            o[IN_OFF + i_x * IN_D + i_y] += cache[IN_OFF + i_x * IN_D + i_y] * cur_val;
        }
}