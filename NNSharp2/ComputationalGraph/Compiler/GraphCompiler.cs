using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.ComputationalGraph.Compiler
{
    public class GraphCompiler
    {
        static Dictionary<MathTypeBase, MathTypeBase> trees;

        static GraphCompiler()
        {
            trees = new Dictionary<MathTypeBase, MathTypeBase>();
        }

        internal static void Add(MathTypeBase outputVariable, MathTypeBase val)
        {
            trees[outputVariable] = val;
        }

        internal static void Build()
        {
            //TODO: when compiling the graph, clone each tree before optimizing
            //TODO: recursively use the associative property for matrix multiplication to attempt to simplify diagonal matrix multiplications
            //TODO: after simplification, find subtrees that are common and separate them into their own variables
            //TODO: sort the subtrees in breadth first order
            //TODO: convert each expression from tree into list in depth first order
            //TODO: generate gpu code for each list

            for(int i = 0; i < trees.Count; i++)
            {
                var pair = trees.ElementAt(i);
                Console.WriteLine($"{pair.Key} = {pair.Value}");
                Console.WriteLine();
            }
        }
    }
}
