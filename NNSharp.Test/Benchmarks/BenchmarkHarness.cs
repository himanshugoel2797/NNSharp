using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Test.Benchmarks
{
    class BenchmarkHarness
    {
        public long Elapsed { get; set; }

        private int runCnt = 0;
        private string name;
        private Stopwatch stopwatch;
        public BenchmarkHarness(string name)
        {
            this.name = name;
        }

        public void Start()
        {
            stopwatch = Stopwatch.StartNew();
        }

        public void Stop()
        {
            stopwatch.Stop();
            Elapsed += stopwatch.ElapsedTicks / (TimeSpan.TicksPerMillisecond / 1000);
            runCnt++;
        }

        public void Show()
        {
            Console.WriteLine($"\t[{name}] Time Taken: {Elapsed / runCnt} ns");
        }
    }
}
