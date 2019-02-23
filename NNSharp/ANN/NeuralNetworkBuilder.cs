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
                if (layer is FCLayer)
                {
                    layer.SetInputSize(o_szs.Last());
                }
                else if (layer is ICNNLayer)
                {
                    if (layers.Last() is ICNNLayer)
                    {
                        layer.SetInputSize((layers.Last() as ICNNLayer).GetFlatOutputSize());
                        o_sz = layer.GetOutputSize((layers.Last() as ICNNLayer).GetFlatOutputSize());
                    }
                    else
                    {
                        layer.SetInputSize((int)Math.Sqrt(o_szs.Last() / (layer as ICNNLayer).GetInputDepth()));
                        o_sz = layer.GetOutputSize((int)Math.Sqrt(o_szs.Last() / (layer as ICNNLayer).GetInputDepth()));
                    }
                }
                else if (layer is ActivationLayer)
                {
                    layer.SetInputSize(o_szs.Last());
                }
            }
            else
            {
                if (layer is FCLayer)
                {
                    o_sz = layer.GetOutputSize(input_sz);
                    layer.SetInputSize(input_sz);
                }
                else if (layer is ICNNLayer)
                {
                    o_sz = layer.GetOutputSize((int)Math.Sqrt(input_sz / (layer as ConvLayer).GetInputDepth()));
                    layer.SetInputSize((int)Math.Sqrt(input_sz / (layer as ConvLayer).GetInputDepth()));
                }
                else
                {
                    o_sz = layer.GetOutputSize(input_sz);
                    layer.SetInputSize(input_sz);
                }
            }

            if (o_sz == 0)
                throw new Exception();

            o_szs.Add(o_sz);
            layers.Add(layer);
            return this;
        }


        public NeuralNetworkBuilder AddPooling(int stride, int filter, int inputDepth)
        {
            var act_layer = new PoolingLayer(stride, filter, inputDepth);
            return Add(act_layer);
        }

        public NeuralNetworkBuilder AddActivation<T>() where T : IActivationFunction, new()
        {
            var act_func = new T();
            var act_layer = new ActivationLayer(act_func);
            return Add(act_layer);
        }

        public NeuralNetworkBuilder AddFC(int output_cnt)
        {
            var fclayer = new FCLayer(output_cnt);
            return Add(fclayer);
        }

        public NeuralNetworkBuilder AddConv(int filter_side, int filter_cnt, int stride = 1, int padding = 0, int input_side = 0, int input_depth = 1)
        {
            var cnvlayer = new ConvLayer();

            cnvlayer.SetFilterCount(filter_cnt);
            cnvlayer.SetFilterSize(filter_side);
            cnvlayer.SetPaddingSize(padding);
            cnvlayer.SetStrideLength(stride);
            cnvlayer.SetInputDepth(input_depth);

            return Add(cnvlayer);
        }

        public NeuralNetworkBuilder WeightInitializer(IWeightInitializer w)
        {
            this.weightInitializer = w;
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
            {
                Console.WriteLine("[WARNING] Weight Initializer not specified.");
            }

            if (lossFunction == null)
                throw new Exception();

            for (int i = 0; i < layers.Count; i++)
            {
                if (weightInitializer != null && layers[i] is IWeightInitializable)
                    (layers[i] as IWeightInitializable).SetWeights(weightInitializer);
            }

            return new NeuralNetwork(layers, input_sz, lossFunction);
        }
    }
}
