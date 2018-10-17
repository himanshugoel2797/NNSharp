using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3.AGNN
{
    public class GeneticTrainer
    {
        private NeuralNetwork network;
        private int seed;
        private Random rng;
        private Genome[] pop;
        private int cur_pop_sz;

        public GeneticTrainer(NeuralNetwork network, int seed, int pop_size)
        {
            pop = new Genome[pop_size];
            this.seed = seed;
            rng = new Random(seed);
            this.network = network;
        }

        public void Train(float[][] inputs, float[][] outputs, int fitness_test_samples, int start_pop_sz, int mating_pop_sz, int child_cnt, float minLoss, float winningLoss, int gens_without_improvement)
        {
            //Initialize start_pop_sz number of genomes
            for (int i = 0; i < start_pop_sz; i++)
            {
                pop[i] = new Genome(network.LayerCount, 10000);
                pop[i].Initialize(rng.Next());
            }
            cur_pop_sz = start_pop_sz;

            float lowest_loss = float.MaxValue;
            int gen = 0, gens_since_improvement = 0;


            while (lowest_loss > winningLoss)
            {
                //Pick fitness_test_samples number of input/output pair
                var fitness_sample_idxs = new int[fitness_test_samples];
                if (fitness_test_samples < inputs.Length)
                {
                    for (int i = 0; i < fitness_test_samples; i++)
                        fitness_sample_idxs[i] = rng.Next() % inputs.Length;
                }
                else
                {
                    for (int i = 0; i < fitness_test_samples; i++)
                        fitness_sample_idxs[i] = i % inputs.Length;
                }

                //Apply each genome{
                for (int i = 0; i < pop.Length; i++)
                {
                    if (pop[i] == null)
                    {
                        cur_pop_sz = i;
                        break;
                    }

                    network.ApplyGenome(pop[i]);

                    pop[i].Loss = 0;

                    //Forward propagate and determine loss
                    for (int j = 0; j < fitness_test_samples; j++)
                    {
                        network.Forward(inputs[fitness_sample_idxs[j]]);
                        //Add the loss values
                        pop[i].Loss += network.Loss(outputs[fitness_sample_idxs[j]]);
                    }

                    float rand = (float)rng.NextGaussian(0, 0.005f);
                    //Console.WriteLine($"Rand = {rand}");
                    //pop[i].Loss += rand;
                    //pop[i].Loss *= (float)System.Math.Exp(-pop[i].NodeLen);
                    pop[i].Loss /= fitness_test_samples;

                    Console.WriteLine($"Gen {gen},Genome [{i}], Loss:{pop[i].Loss}");

                    //Kill and replace any genomes with loss above minLoss and reevaluate
                    /*if (pop[i].Loss > minLoss)
                    {
                        pop[i] = new Genome(network.LayerCount, 10000);
                        pop[i].Initialize(rng.Next());
                        i--;
                    }*/
                }
                //}
                //Sort the genomes by ascending loss values
                Array.Sort(pop, (a, b) =>
                {
                    float a_v = (a == null) ? float.MaxValue : (float)(a.Loss);
                    float b_v = (b == null) ? float.MaxValue : (float)(b.Loss);
                    return a_v.CompareTo(b_v);
                });

                if (lowest_loss > pop[0].Loss)
                {
                    gens_since_improvement = 0;
                    lowest_loss = pop[0].Loss;
                }
                else
                {
                    gens_since_improvement++;
                }

                //For each of the mating_pop_sz number of genomes, generate child_cnt number of children, killing the highest loss genomes to make space{
                //Determine how many low scoring individuals need to be killed for the new individuals
                int net_child_cnt = cur_pop_sz + mating_pop_sz * child_cnt;
                int death_cnt = 0;
                if (net_child_cnt > pop.Length)
                    death_cnt = net_child_cnt - pop.Length;

                //Kill off enough low scoring individuals to make space
                int cur_parent = 0;
                int cur_parent_child_cnt = 0;
                for (int i = cur_pop_sz - death_cnt; i < pop.Length; i++)
                {
                    pop[i] = pop[cur_parent].Mutate();
                    cur_parent_child_cnt++;

                    if (cur_parent_child_cnt == child_cnt)
                    {
                        cur_parent_child_cnt = 0;
                        cur_parent++;
                    }

                    if (cur_parent == mating_pop_sz)
                    {
                        break;
                    }
                }

                Console.WriteLine($"[{gen}]Lowest Loss = {lowest_loss}, Generation Winner = {pop[0].Loss} Node Len = {pop[0].NodeLen}");
                if (gens_since_improvement >= gens_without_improvement)
                    break;

                gen++;
            }

            network.ApplyGenome(pop[0]);
        }
    }
}
