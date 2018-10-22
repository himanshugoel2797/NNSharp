using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SMath = System.Math;

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
        NN,
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

        private double mean, bias, sigma;

        private List<LayerDef> Layers;
        private struct LayerDef
        {
            public LayerType layerType;
            public ActivationFunction activationFunction;
            public int inputSize;
            public int outputSize;
            public NeuralNetwork obj;
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

        public NeuralNetworkBuilder SetWeightInitializer(WeightInitializer weightInitializer, double mean, double sigma, double bias)
        {
            this.mean = mean;
            this.bias = bias;
            this.sigma = sigma;

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

        public NeuralNetworkBuilder Add(NeuralNetwork a)
        {
            LayerDef def = new LayerDef()
            {
                layerType = LayerType.NN,
                inputSize = a.InputSize,
                outputSize = a.OutputSize,
                obj = a,
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
                    break;
            }

            var nn = new NeuralNetwork();
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].layerType == LayerType.NN)
                {
                    nn.Add(Layers[i].obj);
                    continue;
                }

                //Generate matrices + vectors for each layer
                Matrix w = new Matrix(Layers[i].outputSize, Layers[i].inputSize);
                Matrix b = new Matrix(Layers[i].outputSize, 1);

                Matrix o = new Matrix(Layers[i].outputSize, 1);
                Matrix a = new Matrix(Layers[i].outputSize, 1);
                Matrix err = new Matrix(Layers[i].outputSize, 1);

                //Matrix w_shdw = new Matrix(Layers[i].outputSize, Layers[i].inputSize);
                //Matrix b_shdw = new Matrix(1, Layers[i].outputSize);

                var defines = new string[8];
                defines[0] = $"#define ACTIV_FN_IDX ({(int)Layers[i].activationFunction})";
                defines[1] = $"#define O_SZ ({Layers[i].outputSize})";
                defines[2] = $"#define I_SZ ({Layers[i].inputSize})"; 
                defines[3] = $"#define LOSS_FN_IDX ({(int)LossFunction})";

                defines[4] = $"#define X (1)";
                defines[5] = $"#define Y (1)";
                defines[6] = $"#define Z (1)";
                defines[7] = $"#define F(row, col) (row * I_SZ + col)";

                var n = new float[2];
                //defines[3] = $"#define OPT_FN_IDX ({(int)LossFunction})";
                switch (WeightInitializer)
                {
                    case WeightInitializer.UniformNoise:
                        {
                            weight_init = Shader.FromFile("uniform_weight_init.glsl", $"#define MEAN ({mean})", $"#define SIGMA ({SMath.Sqrt(6.0f / (Layers[i].inputSize + Layers[i].outputSize))})", $"#define BIAS ({bias})", defines[1], defines[2], defines[4], defines[5], defines[6], defines[7]);
                            weight_init.Set("w", w.tex, false, true);
                            weight_init.Set("b", b.tex, false, true);
                            weight_init.Set("seed0", (float)rng.NextDouble());
                            weight_init.Set("seed1", (float)rng.NextDouble());

                            weight_init.Dispatch((uint)w.tex.Width, (uint)w.tex.Height, 1);
                            weight_init.Dispose();
                        }
                        break;
                }

                //Generate shaders for each layer
                Shader fwd = null;
                Shader bkwd = null;
                Shader bkwd_wu = null;
                Shader bkwd_werr = null;

                switch (Layers[i].layerType)
                {
                    case LayerType.FC:
                        {
                            fwd = Shader.FromFile("fc_fwd.glsl", defines);
                            if (i == Layers.Count - 1)
                                bkwd = Shader.FromFile("fc_bkwd_last.glsl", defines);
                            else
                            { 
                                bkwd = Shader.FromFile("fc_bkwd.glsl", defines);
                                bkwd_werr = Shader.FromFile("fc_bkwd_werr.glsl", defines);
                            }

                            bkwd_wu = Shader.FromFile("fc_bkwd_wu.glsl", defines);
                        }
                        break;
                }

                nn.Add(Layers[i].layerType, new Matrix[] { o, a, err }, new Matrix[] { w, b }, new Matrix[] { /*w_shdw, b_shdw*/ }, fwd, bkwd_werr, bkwd, bkwd_wu);
            }

            return nn;
        }
    }
}
