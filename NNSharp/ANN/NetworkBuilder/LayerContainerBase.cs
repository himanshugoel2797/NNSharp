using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.NetworkBuilder
{
    [Serializable]
    public abstract class LayerContainerBase
    {
        protected IList<LayerContainerBase> InputLayers;
        protected IList<LayerContainerBase> OutputLayers;
        
        public LayerContainerBase()
        {
            InputLayers = new List<LayerContainerBase>();
            OutputLayers = new List<LayerContainerBase>();
        }

        #region Internal State Update/Checking
        protected internal virtual void AddInputLayer(LayerContainerBase input)
        {
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
        public virtual LayerContainerBase Append(params LayerContainerBase[] layer)
        {
            for (int i = 0; i < layer.Length; i++)
            {
                OutputLayers.Add(layer[i]);
                layer[i].AddInputLayer(this);
            }
            return this;
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
        public virtual void SetupInternalState()
        {
            for (int i = 0; i < OutputLayers.Count; i++)
                OutputLayers[i].SetupInternalState();
        }
        #endregion

        #region Forward Propagation
        public abstract Vector[] Forward(params Vector[] input);
        public abstract Vector[] ForwardPropagate(params Vector[] input);
        #endregion

        #region Backward Propagation
        #region Gradients Between Layers
        public abstract Vector[] ComputeGradients(params Vector[] prev_delta);
        #endregion

        #region Gradients Per Layer
        public abstract void ComputeLayerErrors(params Vector[] prev_delta);
        public abstract void ResetLayerErrors();
        public abstract void UpdateLayers(IOptimizer optimizer);
        #endregion
        #endregion

        #region Save/Load
        public void Save(string file)
        {
            var serializer = new BinaryFormatter();
            using (FileStream t = File.Create(file))
                serializer.Serialize(t, this);
        }

        public static LayerContainerBase Load(string file)
        {
            LayerContainerBase n = null;
            var serializer = new BinaryFormatter();

            using (FileStream t = File.OpenRead(file))
                n = (LayerContainerBase)serializer.Deserialize(t);

            n.SetupInternalState();
            return n;
        }
        #endregion
    }
}
