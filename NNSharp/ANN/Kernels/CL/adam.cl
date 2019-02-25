//LEN

__kernel void adam(const float learning_rate,
                   const float beta_1,
                   const float beta_2,
                   __global float4* m,
                   __global float4* v,
                   const __global float4* nabla,
                   __global float4* o
                         ){
        const int g_row = get_global_id(0) * WPT; // WPT = Work per thread

        for(int w = 0; w < WPT; w++)
                if(g_row + w < LEN){
                        //const float cond = float(isless(g_row + w, LEN));

                        const float4 n = nabla[g_row + w];
                        float4 m_w = beta_1 * m[g_row + w] + (1 - beta_1) * n;
                        float4 v_w = beta_2 * v[g_row + w] + (1 - beta_2) * n * n;
                        
                        o[g_row + w] -= (learning_rate / (sqrt(v_w / (1 - beta_2)) + 1e-10f)) * (m_w / (1 - beta_1));
                        m[g_row + w] = m_w;
                        v[g_row + w] = v_w;
                }
}