using NNSharp.ANN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN
{
    public class NeuralNetworkBuilder
    {
        private List<ILayer> layers;
        private List<int> o_szs;
        private int input_sz;

        private IWeightInitializer weightInitializer;
        private ILossFunction lossFunction;
        private IOptimizer optimizer;

        public NeuralNetworkBuilder(int input_sz)
        {
            layers = new List<ILayer>();
            o_szs = new List<int>();

            this.input_sz = input_sz;
        }

        public NeuralNetworkBuilder Add(ILayer layer)
        {
            int o_sz;
            if (o_szs.Count > 0)
            {
                o_sz = layer.GetOutputSize(o_szs.Last());
                layer.SetInputSize(o_szs.Last());
            }
            else
            {
                o_sz = layer.GetOutputSize(input_sz);
                layer.SetInputSize(input_sz);
            }

            if (o_sz == 0)
                throw new Exception();

            o_szs.Add(o_sz);
            layers.Add(layer);
            return this;
        }

        public NeuralNetworkBuilder AddFC<T>(int output_cnt) where T : IActivationFunction, new()
        {
            var fclayer = new FCLayer(output_cnt, new T());
            return Add(fclayer);
        }

        public NeuralNetworkBuilder WeightInitializer(IWeightInitializer w)
        {
            this.weightInitializer = w;
            return this;
        }

        public NeuralNetworkBuilder Optimizer(IOptimizer o, float rate)
        {
            this.optimizer = o;
            optimizer.SetLearningRate(rate);
            return this;
        }

        public NeuralNetworkBuilder LossFunction<T>() where T : ILossFunction, new()
        {
            lossFunction = new T();
            return this;
        }

        public NeuralNetwork Build()
        {
            if (layers.Count == 0)
                throw new Exception();

            if (weightInitializer == null)
                throw new Exception();

            if (lossFunction == null)
                throw new Exception();

            if (optimizer == null)
                throw new Exception();

            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].SetOptimizer(optimizer);
                if (layers[i] is IWeightInitializable)
                    (layers[i] as IWeightInitializable).SetWeights(weightInitializer);
            }

            return new NeuralNetwork(layers, input_sz, lossFunction);
        }
    }
}
