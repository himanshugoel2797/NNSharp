using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.ANN
{
    public enum LossFunction
    {
        MeanSquaredError,
    }

    public enum WeightInitializer
    {
        UniformNoise,
    }

    public enum ActivationFunction
    {
        Sigmoid,
        Tanh,
        ReLU,
    }

    public enum LayerType
    {
        FC,
        Conv,
        DeConv,
    }

    public class NeuralNetworkBuilder
    {
        public int InputSize { get; private set; }
        public LossFunction LossFunction { get; private set; }
        public WeightInitializer WeightInitializer { get; private set; }
        public IOptimizer Optimizer { get; private set; }

        private double mean, bias;

        private List<LayerDef> Layers;
        private struct LayerDef
        {
            public LayerType layerType;
            public ActivationFunction activationFunction;
            public int inputSize;
            public int outputSize;
        }

        public NeuralNetworkBuilder(int inputSize)
        {
            InputSize = inputSize;
            Layers = new List<LayerDef>();
        }

        public NeuralNetworkBuilder SetLossFunction(LossFunction loss)
        {
            LossFunction = loss;
            return this;
        }

        public NeuralNetworkBuilder SetWeightInitializer(WeightInitializer weightInitializer, double mean, double bias)
        {
            this.mean = mean;
            this.bias = bias;

            WeightInitializer = weightInitializer;
            return this;
        }

        public NeuralNetworkBuilder SetOptimizer(IOptimizer optimizer)
        {
            Optimizer = optimizer;
            return this;
        }

        public NeuralNetworkBuilder AddFCLayer(int output_sz, ActivationFunction activ)
        {
            LayerDef def = new LayerDef()
            {
                activationFunction = activ,
                outputSize = output_sz,
                layerType = LayerType.FC,
                inputSize = (Layers.Count > 0) ? Layers.Last().outputSize : InputSize
            };

            Layers.Add(def);
            return this;
        }

        public NeuralNetwork Build()
        {
            Random rng = new Random(0);
            Shader weight_init = null;

            switch (WeightInitializer)
            {
                case WeightInitializer.UniformNoise:
                    weight_init = Shader.FromFile("uniform_weight_init.glsl", $"#define MEAN ({mean})", $"#define BIAS ({bias})");
                    break;
            }

            var nn = new NeuralNetwork();
            for (int i = 0; i < Layers.Count; i++)
            {
                //Generate matrices + vectors for each layer
                Matrix w = new Matrix(Layers[i].outputSize, Layers[i].inputSize);
                Matrix b = new Matrix(1, Layers[i].outputSize);

                Matrix o = new Matrix(1, Layers[i].outputSize);
                Matrix a = new Matrix(1, Layers[i].outputSize);

                Matrix w_shdw = new Matrix(Layers[i].outputSize, Layers[i].inputSize);
                Matrix b_shdw = new Matrix(1, Layers[i].outputSize);
                
                int i_sz = Layers[i].inputSize;
                if (i_sz % 4 != 0)
                    i_sz += 4 - (i_sz % 4);
                 
                var defines = new string[7];
                defines[0] = $"#define ACTIV_FN_IDX ({(int)Layers[i].activationFunction})";
                defines[1] = $"#define O_SZ ({Layers[i].outputSize})";
                defines[2] = $"#define I_SZ ({i_sz})";
                defines[3] = $"#define LOSS_FN_IDX ({(int)LossFunction})";

                defines[4] = $"#define X (1)";
                defines[5] = $"#define Y (1)";
                defines[6] = $"#define Z (1)";

                //defines[3] = $"#define OPT_FN_IDX ({(int)LossFunction})";

                switch (WeightInitializer)
                {
                    case WeightInitializer.UniformNoise:
                        {
                            weight_init.Set("w", w.tex, false, true);
                            weight_init.Set("b", b.tex, false, true);
                            weight_init.Set("seed0", (float)rng.NextDouble());
                            weight_init.Set("seed1", (float)rng.NextDouble());
                            weight_init.Set("cols_cnt", w.Columns);
                            weight_init.Set("O_SZ", w.Rows);

                            weight_init.Dispatch((uint)w.tex.Width, (uint)w.tex.Height, 1);
                        } 
                        break;
                }

                //Generate shaders for each layer
                Shader fwd = null;
                Shader bkwd = null;

                switch (Layers[i].layerType)
                {
                    case LayerType.FC:
                        {
                            defines[4] = $"#define X (4)";

                            fwd = Shader.FromFile("fc_fwd.glsl", defines);
                            if (i == Layers.Count - 1)
                                bkwd = Shader.FromFile("fc_bkwd_last.glsl", defines);
                            else
                                bkwd = Shader.FromFile("fc_bkwd.glsl", defines);
                        }
                        break;
                }

                nn.Add(Layers[i].layerType, new Matrix[] { o, a }, new Matrix[] { w, b }, new Matrix[] { w_shdw, b_shdw }, fwd, bkwd);
            }

            return nn;
        }
    }
}
