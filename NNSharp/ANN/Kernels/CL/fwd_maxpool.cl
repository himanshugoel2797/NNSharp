//IN_D
//KERN_D
//OUT_D
//STRIDE

__kernel void fwd_maxpool(const int IN_OFF,
                          const int OUT_OFF,
                          const __global float* i,
                          __global float* cache,
                          __global float* o) {

    const int globalRow = get_global_id(0); //row
    const int globalCol = get_global_id(1); //col

    //Output position    
    const int row = globalRow;
    const int col = globalCol;

    float acc = FLT_MIN;
    int off = 0;

    for (int n0 = 0; n0 < KERN_D; n0++)
        for (int n1 = 0; n1 < KERN_D; n1++)
        {
            int i_x = col * STRIDE + (n0 - KERN_D / 2) + KERN_D / 2;
            int i_y = row * STRIDE + (n1 - KERN_D / 2) + KERN_D / 2;

            const float i_val = i[IN_OFF + i_x * IN_D + i_y];

            cache[IN_OFF + i_x * IN_D + i_y] = 0;
            if(i_val > acc){
                acc = i_val;
                off = IN_OFF + i_x * IN_D + i_y;
            }
        }

    cache[off] = 1;
    o[OUT_OFF + col * OUT_D + row] = acc;
}