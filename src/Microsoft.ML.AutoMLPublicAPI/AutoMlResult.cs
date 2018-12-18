using System.Collections.Generic;
using Microsoft.ML.Runtime.PipelineInference;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class AutoMlResult
    {
        public IReadOnlyList<IterationResult> IterationResults { get; private set; }
        public ITransformer BestIterationModel { get; private set; }

        internal AutoMlResult(IReadOnlyList<IterationResult> iterationResults, ITransformer bestIterationModel)
        {
            IterationResults = iterationResults;
            BestIterationModel = bestIterationModel;
        }

        public IterationResult BestIteration
        {
            get
            {
                return IterationResults.OrderByDescending(r => r.Score.RSquared).First();
            }
        }
    }
}
