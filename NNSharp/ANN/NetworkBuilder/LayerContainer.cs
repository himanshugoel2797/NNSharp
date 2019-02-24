using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.NetworkBuilder
{
    public class LayerContainer : LayerContainerBase
    {
        protected ILayer CurrentLayer;
        private bool isWeightInitialized;

        public LayerContainer(ILayer layer) : base()
        {
            CurrentLayer = layer;
            isWeightInitialized = false;
        }

        #region Internal State Update/Checking
        protected internal override void Initializer(LayerContainerBase input)
        {
            if (input is LayerContainer)
                CurrentLayer.SetInputSize((input as LayerContainer).CurrentLayer.GetOutputSize(), (input as LayerContainer).CurrentLayer.GetOutputDepth());
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
        public override Vector[] Forward(Vector[] input)
        {
            return CurrentLayer.Forward(input);
        }

        public override Vector[] ForwardPropagate(Vector[] input)
        {
            List<Vector> rets = new List<Vector>();
            var result = Forward(input);
            for (int i = 0; i < OutputLayers.Count; i++)
                rets.AddRange(OutputLayers[i].ForwardPropagate(result));
            return rets.ToArray();
        }
        #endregion

        #region Backward Propagation
        #region Gradients Between Layers
        public override Vector[] ComputeGradients(Vector[] prev_delta)
        {
            List<Vector> rets = new List<Vector>();
            var delta = CurrentLayer.Propagate(prev_delta);
            for (int i = 0; i < InputLayers.Count; i++)
                rets.AddRange(InputLayers[i].ComputeGradients(delta));
            return rets.ToArray();
        }
        #endregion

        #region Gradients Per Layer
        public override void ComputeLayerErrors(Vector[] prev_delta)
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
