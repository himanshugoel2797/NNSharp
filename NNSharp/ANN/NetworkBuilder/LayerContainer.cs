using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.NetworkBuilder
{
    [Serializable]
    public class LayerContainer : LayerContainerBase
    {
        public ILayer CurrentLayer { get; private set; }
        private bool isWeightInitialized;

        public LayerContainer(ILayer layer) : base()
        {
            CurrentLayer = layer;
            isWeightInitialized = false;
        }

        #region Internal State Update/Checking
        public override void SetupInternalState()
        {
            int input_sz = 0, input_dpth = 0;
            for (int i = 0; i < InputLayers.Count; i++)
                if (InputLayers[i] is LayerContainer)
                {
                    input_sz = (InputLayers[i] as LayerContainer).CurrentLayer.GetOutputSize();
                    input_dpth = (InputLayers[i] as LayerContainer).CurrentLayer.GetOutputDepth();
                }
            CurrentLayer.SetInputSize(input_sz, input_dpth);
            base.SetupInternalState();
        }

        public override void Check()
        {
            if (InputLayers.Count > 0)
            {
                /*
                int input_sz = InputLayers[0].CurrentLayer.GetOutputSize();
                int input_dpth = InputLayers[0].CurrentLayer.GetOutputDepth();

                for (int i = 1; i < InputLayers.Count; i++)
                    if (InputLayers[i].CurrentLayer.GetOutputSize() != input_sz | InputLayers[i].CurrentLayer.GetOutputDepth() != input_dpth)
                        throw new Exception("Input dimensions don't match.");
                        */
            }
        }
        #endregion

        #region Weight Initialization
        public void InitializeWeights(IWeightInitializer weightInitializer)
        {
            if (!isWeightInitialized)
            {
                isWeightInitialized = true;
                if (CurrentLayer is IWeightInitializable)
                    (CurrentLayer as IWeightInitializable).SetWeights(weightInitializer);
            }

            for (int i = 0; i < OutputLayers.Count; i++)
                if (OutputLayers[i] is LayerContainer)
                    (OutputLayers[i] as LayerContainer).InitializeWeights(weightInitializer);
        }
        #endregion

        #region Forward Propagation
        public override Matrix[] Forward(params Matrix[] input)
        {
            return CurrentLayer.Forward(input);
        }

        public override Matrix[] ForwardPropagate(params Matrix[] input)
        {
            List<Matrix> rets = new List<Matrix>();
            var result = Forward(input);
            if (OutputLayers.Count == 0)
                return result;

            for (int i = 0; i < OutputLayers.Count; i++)
                rets.AddRange(OutputLayers[i].ForwardPropagate(result));
            return rets.ToArray();
        }
        #endregion

        #region Backward Propagation
        #region Gradients Between Layers
        public override Matrix[] ComputeGradients(params Matrix[] prev_delta)
        {
            List<Matrix> rets = new List<Matrix>();
            var delta = CurrentLayer.Propagate(prev_delta);
            if (InputLayers.Count == 0)
                return delta;

            for (int i = 0; i < InputLayers.Count; i++)
                rets.AddRange(InputLayers[i].ComputeGradients(delta));
            return rets.ToArray();
        }
        #endregion

        #region Gradients Per Layer
        public override void ComputeLayerErrors(params Matrix[] prev_delta)
        {
            CurrentLayer.LayerError(prev_delta);
            var delta = CurrentLayer.GetLastDelta();
            for (int i = 0; i < InputLayers.Count; i++)
                InputLayers[i].ComputeLayerErrors(delta);
        }

        public override void ResetLayerErrors()
        {
            CurrentLayer.ResetLayerError();
            for (int i = 0; i < InputLayers.Count; i++)
                InputLayers[i].ResetLayerErrors();
        }

        public override void UpdateLayers(IOptimizer optimizer)
        {
            CurrentLayer.Learn(optimizer);
            for (int i = 0; i < InputLayers.Count; i++)
                InputLayers[i].UpdateLayers(optimizer);
        }
        #endregion
        #endregion
    }
}
