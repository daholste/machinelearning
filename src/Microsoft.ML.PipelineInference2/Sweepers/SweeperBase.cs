// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.Sweeper;

namespace Microsoft.ML.Runtime.Sweeper
{
    /// <summary>
    /// Signature for the GUI loaders of sweepers.
    /// </summary>
    public delegate void SignatureSweeperFromParameterList(IValueGenerator[] sweepParameters);

    /// <summary>
    /// Base sweeper that ensures the suggestions are different from each other and from the previous runs.
    /// </summary>
    public abstract class SweeperBase : ISweeper
    {
        public class ArgumentsBase
        {
            //[Argument(ArgumentType.Multiple, HelpText = "Swept parameters", ShortName = "p", SignatureType = typeof(SignatureSweeperParameter))]
            public IComponentFactory<IValueGenerator>[] SweptParameters;

            //[Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of tries to generate distinct parameter sets.", ShortName = "r")]
            public int Retries = 10;
        }

        private readonly ArgumentsBase _args;
        protected readonly IValueGenerator[] SweepParameters;
        protected readonly IHost Host;

        protected SweeperBase(ArgumentsBase args, IHostEnvironment env, string name)
        {
            //Contracts.CheckValue(env, nameof(env));
            //env.CheckNonWhiteSpace(name, nameof(name));
            //Host = env.Register(name);
            //Host.CheckValue(args, nameof(args));
            //Host.CheckNonEmpty(args.SweptParameters, nameof(args.SweptParameters));

            _args = args;

            SweepParameters = args.SweptParameters.Select(p => p.CreateComponent(Host)).ToArray();
        }

        protected SweeperBase(ArgumentsBase args, IHostEnvironment env, IValueGenerator[] sweepParameters, string name)
        {
            //Contracts.CheckValue(env, nameof(env));
            //env.CheckNonWhiteSpace(name, nameof(name));
            //Host = env.Register(name);
            //Host.CheckValue(args, nameof(args));
            //Host.CheckValue(sweepParameters, nameof(sweepParameters));

            _args = args;
            SweepParameters = sweepParameters;
        }

        public virtual ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            var prevParamSets = previousRuns?.Select(r => r.ParameterSet).ToList() ?? new List<ParameterSet>();
            var result = new HashSet<ParameterSet>();
            for (int i = 0; i < maxSweeps; i++)
            {
                ParameterSet paramSet;
                int retries = 0;
                do
                {
                    paramSet = CreateParamSet();
                    ++retries;
                } while (paramSet != null && retries < _args.Retries &&
                    (AlreadyGenerated(paramSet, prevParamSets) || AlreadyGenerated(paramSet, result)));

                AutoMlUtils.Assert(paramSet != null);
                result.Add(paramSet);
            }

            return result.ToArray();
        }

        protected abstract ParameterSet CreateParamSet();

        protected static bool AlreadyGenerated(ParameterSet paramSet, IEnumerable<ParameterSet> previousRuns)
        {
            return previousRuns.Any(previousRun => previousRun.Equals(paramSet));
        }
    }
}
