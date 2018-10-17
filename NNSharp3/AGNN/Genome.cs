using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.AGNN
{
    public class Genome
    {
        public int[][] Nodes;
        public int NodeLen;

        public int Seed;
        public int MaxLength;

        public float Loss;

        private Random rng;

        public Genome(int nodeCnt, int maxLen)
        {
            Nodes = new int[nodeCnt][];
            NodeLen = 0;

            MaxLength = maxLen;
        }

        public Genome(Genome parent)
        {
            Nodes = new int[parent.Nodes.Length][];
            NodeLen = parent.NodeLen;
            rng = new Random(parent.rng);
            MaxLength = parent.MaxLength;

            for(int i = 0; i < Nodes.Length; i++)
            {
                Nodes[i] = new int[NodeLen + 1];

                for (int j = 0; j < NodeLen; j++)
                    Nodes[i][j] = parent.Nodes[i][j];

                Nodes[i][NodeLen] = rng.Next();
            }
            NodeLen++;
        }

        public void Initialize()
        {
            Random rng = new Random();
            Initialize(rng.Next());
        }

        public void Initialize(int seed)
        {
            Seed = seed;
            rng = new Random(seed);

            for (int i = 0; i < Nodes.Length; i++)
            {
                Nodes[i] = new int[1];
                Nodes[i][0] = rng.Next();
            }

            NodeLen = 1;
        }

        public Genome Mutate()
        {
            if (NodeLen < MaxLength)
            {
                return new Genome(this);
            }
            else
                throw new Exception();
        }
    }
}
