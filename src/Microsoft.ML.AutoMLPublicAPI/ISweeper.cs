using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Sweeper.Algorithms;
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Sweeper;
using System.Diagnostics;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class MySweeper
    {

        public static void SweepNow(MLContext context)
        {
            SmacSweeper sweeper = new SmacSweeper(context, new SmacSweeper.Arguments()
            {
                SweptParameters = new IComponentFactory<INumericValueGenerator>[] {
                                ComponentFactoryUtils.CreateFromFunction(
                                    t => new FloatValueGenerator(new FloatParamArguments() { Name = "foo", Min = 1, Max = 5})),
                                ComponentFactoryUtils.CreateFromFunction(
                                    t => new LongValueGenerator(new LongParamArguments() { Name = "bar", Min = 1, Max = 1000, LogBase = true }))
                            },
            });

            Random rand = new Random();
            List<RunResult> results = new List<RunResult>();
            for (int i=0; i<11; i++)
            {
                float foo = rand.Next(1,5);
                long bar = rand.Next(1,1000);
                ParameterSet p = new ParameterSet(new List<IParameterValue>() { new FloatParameterValue("foo", foo), new LongParameterValue("bar", bar) } );
                Double metric = ((5 - Math.Abs(4 - foo)) * 200) + (1001 - Math.Abs(33 - bar)) + rand.Next(1, 20);
                Debug.Assert(metric > 0);
                results.Add(new RunResult(p, metric, true));
            }
            int count = 0;
            while (true)
            {
                ParameterSet[] pars = sweeper.ProposeSweeps(1, results);
                foreach(ParameterSet p in pars)
                {
                    float foo = 0;
                    long bar = 0;

                    foo = (p["foo"] as FloatParameterValue).Value;
                    bar = (p["bar"] as LongParameterValue).Value;

                    Double metric = ((5 - Math.Abs(4 - foo)) * 200) + (1001 - Math.Abs(33 - bar)) + rand.Next(1, 20);
                    results.Add(new RunResult(p, metric, true));
                    count++;
                    Console.WriteLine("{0}--{1}--{2}--{3}", count, foo, bar, metric);
                }

                // Console.ReadLine();
            }
        }
    }
}
