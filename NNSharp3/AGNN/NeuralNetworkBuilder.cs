using NNSharp3.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.AGNN
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
        NN,
    }

    public class NeuralNetworkBuilder
    {
        public int InputSize { get; private set; }
        public LossFunction LossFunction { get; private set; }

        private double mean, bias;

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
                outputSize = a.OutputSize,
                layerType = LayerType.FC,
                inputSize = a.InputSize,
                obj = a,
            };

            Layers.Add(def);
            return this;
        }

        public NeuralNetwork Build()
        {
            Random rng = new Random(0);

            var nn = new NeuralNetwork();
            for (int i = 0; i < Layers.Count; i++)
            {
                //Generate matrices + vectors for each layer
                switch (Layers[i].layerType)
                {
                    case LayerType.FC:
                        {
                            var w = new float[Layers[i].outputSize * Layers[i].inputSize];
                            var b = new float[Layers[i].outputSize];

                            var o = new float[Layers[i].outputSize];
                            var a = new float[Layers[i].outputSize];

                            nn.Add(Layers[i].layerType, Layers[i].activationFunction, new float[][] { o, a }, new float[][] { w, b });
                        }
                        break;
                    case LayerType.NN:
                        {
                            nn.Add(Layers[i].obj);
                        }
                        break;
                }
            }

            return nn;
        }
    }
}
