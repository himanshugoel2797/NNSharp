﻿using NNSharp.ANN.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Optimizers
{
    [Serializable]
    public class Adam : IOptimizer
    {
        private class AdamParams
        {
            public Matrix[] m_w;
            public Matrix[] v_w;

            public Matrix[] m_b;
            public Matrix[] v_b;
        }

        private readonly float learning_rate;
        private readonly float beta_1;
        private readonly float beta_2;
        private readonly float epsilon;
        private Dictionary<ILayer, AdamParams> layers;

        public float L1Val { get; private set; }
        public float L2Val { get; private set; }
        public int Net { get; private set; }

#if GPU
        private static Dictionary<int, Kernel> adam_kernels;

        static Adam()
        {
            adam_kernels = new Dictionary<int, Kernel>();
        }

        private void Optimize(Memory m, Memory v, Memory nabla, Memory o, int o_len)
        {
            var dev = Device.GetDevice();

            while (o_len % 4 != 0)
                o_len++;

            int len = o_len / 4;
            var wpt = (KernelManager.MaxWPT - 1);
            while ((1 << wpt > len | len / (1 << wpt) < KernelManager.Ratio) && wpt >= 0)
                wpt--;
            if (wpt < 0) wpt = 0;

            if (!adam_kernels.ContainsKey(len))
            {
                adam_kernels[len] = dev.LoadKernel("adam", "", $"#define WPT ({1 << wpt})", $"#define LEN ({len})");
            }

            adam_kernels[len]
                .SetArgument(learning_rate)
                .SetArgument(beta_1)
                .SetArgument(beta_2)
                .SetArgumentMemory(m)
                .SetArgumentMemory(v)
                .SetArgumentMemory(nabla)
                .SetArgumentMemory(o);

            dev.Dispatch(adam_kernels[len], new uint[] { (uint)(len / (1 << wpt) + 1), 1 }, null);
        }
#endif

        public Adam(float learning_rate = 0.001f, float epsilon = 1e-6f, float beta_1 = 0.9f, float beta_2 = 0.999f)
        {
            this.layers = new Dictionary<ILayer, AdamParams>();
            this.learning_rate = learning_rate;
            this.beta_1 = beta_1;
            this.beta_2 = beta_2;
            this.epsilon = epsilon;
            Net = 1;
        }

        public void OptimizeWeights(ILayer layer, int idx, Matrix w, Matrix nabla_w)
        {
            var @params = layers[layer];

            //m_w = beta_1 * m_w + (1 - beta_1) * nabla_w
            //v_w = beta_2 * v_w + (1 - beta_2) * nabla_w^2
            //w = w - (learning_rate / (sqrt(v_w / (1 - beta_2)) + eps)) * (m_w / (1 - beta_1))
#if CPU
            Parallel.For(0, @params.m_w[idx].Memory.Length, (i) =>
            {
                @params.m_w[idx].Memory[i] = beta_1 * @params.m_w[idx].Memory[i] + (1 - beta_1) * nabla_w.Memory[i];
                @params.v_w[idx].Memory[i] = beta_2 * @params.v_w[idx].Memory[i] + (1 - beta_2) * nabla_w.Memory[i] * nabla_w.Memory[i];

                L1Val += Math.Abs(w.Memory[i]);
                L2Val += w.Memory[i] * w.Memory[i];
                Net++;

                w.Memory[i] -= (float)(learning_rate / (Math.Sqrt(@params.v_w[idx].Memory[i] / (1 - beta_2)) + epsilon)) * (@params.m_w[idx].Memory[i] / (1 - beta_1));
            });
#elif GPU
            Optimize(@params.m_w[idx].memory, @params.v_w[idx].memory, nabla_w.memory, w.memory, w.Width * w.Height);
#endif
        }

        public void OptimizeBiases(ILayer layer, int idx, Matrix b, Matrix nabla_b)
        {
            var @params = layers[layer];

            //m_b = beta_1 * m_b + (1 - beta_1) * nabla_b
            //v_b = beta_2 * v_b + (1 - beta_2) * nabla_b^2
            //b = b - (learning_rate / (sqrt(v_b / (1 - beta_2)) + eps)) * (m_b / (1 - beta_1))
#if CPU
            Parallel.For(0, @params.m_b[idx].Memory.Length, (i) =>
            {
                @params.m_b[idx].Memory[i] = beta_1 * @params.m_b[idx].Memory[i] + (1 - beta_1) * nabla_b.Memory[i];
                @params.v_b[idx].Memory[i] = beta_2 * @params.v_b[idx].Memory[i] + (1 - beta_2) * nabla_b.Memory[i] * nabla_b.Memory[i];

                b.Memory[i] -= (float)(learning_rate / (Math.Sqrt(@params.v_b[idx].Memory[i] / (1 - beta_2)) + epsilon)) * (@params.m_b[idx].Memory[i] / (1 - beta_1));
            });
#elif GPU
            Optimize(@params.m_b[idx].memory, @params.v_b[idx].memory, nabla_b.memory, b.memory, b.Length);
#endif
        }

        public void RegisterLayer(ILayer layer, int w_cnt, int ww_len, int wh_len, int b_cnt, int b_len)
        {
            if (!layers.ContainsKey(layer))
            {
                layers[layer] = new AdamParams()
                {
                    m_w = new Matrix[w_cnt],
                    v_w = new Matrix[w_cnt],
                    m_b = new Matrix[b_cnt],
                    v_b = new Matrix[b_cnt]
                };

                for (int i = 0; i < w_cnt; i++)
                {
                    layers[layer].m_w[i] = new Matrix(ww_len, wh_len, MemoryFlags.ReadWrite, true);
                    layers[layer].v_w[i] = new Matrix(ww_len, wh_len, MemoryFlags.ReadWrite, true);
                }

                for (int i = 0; i < b_cnt; i++)
                {
                    layers[layer].m_b[i] = new Matrix(b_len, 1, MemoryFlags.ReadWrite, true);
                    layers[layer].v_b[i] = new Matrix(b_len, 1, MemoryFlags.ReadWrite, true);
                }
            }
        }

        public void Update(float curError)
        {
            L1Val = 0;
            L2Val = 0;
            Net = 0;
        }
    }
}
