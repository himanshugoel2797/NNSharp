//IN_D
//KERN_D
//IN_P
//OUT_D
//STRIDE

//ROT

__kernel void conv(const int IN_OFF,
                   const int KERN_OFF,
                   const int OUT_OFF,
                   const __global float* i,
                   const __global float* kern,
                   __global float* o) {

    const int globalRow = get_global_id(0); //row
    const int globalCol = get_global_id(1); //col

    //Output position    
    const int row = globalRow;
    const int col = globalCol;

    float acc = 0.0f;

    for (int n0 = 0; n0 < KERN_D; n0++)
        for (int n1 = 0; n1 < KERN_D; n1++)
        {
            int i_x = col * STRIDE + (n0 - KERN_D / 2) + KERN_D / 2 - IN_P;
            int i_y = row * STRIDE + (n1 - KERN_D / 2) + KERN_D / 2 - IN_P;

            if (i_x >= 0 && i_y >= 0 && i_x < IN_D && i_y < IN_D)
#ifdef ROT
                acc += kern[KERN_OFF + (KERN_D - 1 - n0) * KERN_D + (KERN_D - 1 - n1)] * i[IN_OFF + i_x * IN_D + i_y];
#else
                acc += kern[KERN_OFF + n0 * KERN_D + n1] * i[IN_OFF + i_x * IN_D + i_y];
#endif
        }

    o[OUT_OFF + col * OUT_D + row] += acc;
}