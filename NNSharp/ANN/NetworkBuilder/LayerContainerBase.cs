using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.NetworkBuilder
{
    [Serializable]
    public abstract class LayerContainerBase
    {
        protected IList<LayerContainerBase> InputLayers;
        protected IList<LayerContainerBase> OutputLayers;
        private bool isInitialized;

        public LayerContainerBase()
        {
            InputLayers = new List<LayerContainerBase>();
            OutputLayers = new List<LayerContainerBase>();
            isInitialized = false;
        }

        #region Internal State Update/Checking
        protected internal virtual void Initializer(LayerContainerBase input) { }

        protected internal virtual void AddInputLayer(LayerContainerBase input)
        {
            if (!isInitialized)
            {
                Initializer(input);
                isInitialized = true;
            }
            InputLayers.Add(input);
        }

        protected internal virtual void RemoveInputLayer(LayerContainerBase input)
        {
            InputLayers.Remove(input);
        }

        public virtual void Check()
        {
        }
        #endregion

        #region Manage Connections
        public virtual void Append(params LayerContainerBase[] layer)
        {
            for (int i = 0; i < layer.Length; i++)
            {
                OutputLayers.Add(layer[i]);
                layer[i].AddInputLayer(this);
            }
        }

        public virtual void Remove(params LayerContainerBase[] layer)
        {
            for (int i = 0; i < layer.Length; i++)
                if (OutputLayers.Contains(layer[i]))
                {
                    OutputLayers.Remove(layer[i]);
                    layer[i].RemoveInputLayer(this);
                }
        }
        #endregion

        #region Setup Internal State
        public virtual void SetupInternalState() { }
        #endregion

        #region Forward Propagation
        public abstract Vector[] Forward(Vector[] input);
        public abstract Vector[] ForwardPropagate(Vector[] input);
        #endregion

        #region Backward Propagation
        #region Gradients Between Layers
        public abstract Vector[] ComputeGradients(Vector[] prev_delta);
        #endregion

        #region Gradients Per Layer
        public abstract void ComputeLayerErrors(Vector[] prev_delta);
        public abstract void ResetLayerErrors();
        public abstract void UpdateLayers(IOptimizer optimizer);
        #endregion
        #endregion
    }
}
