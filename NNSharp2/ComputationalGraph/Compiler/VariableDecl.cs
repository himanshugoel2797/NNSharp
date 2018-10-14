namespace NNSharp2.ComputationalGraph.Compiler
{
    internal class VariableDecl
    {
        public string Name { get; private set; }
        public bool Constant { get; private set; }
        public bool IsMatrix { get; private set; }

        public int[] Dimensions { get; private set; }
        public double Value { get; private set; }
    }
}