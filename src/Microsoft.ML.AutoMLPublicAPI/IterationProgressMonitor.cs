using System.Collections.Generic;
using Microsoft.ML.Runtime.PipelineInference;
using System.Linq;
using System;

namespace Microsoft.ML.AutoMLPublicAPI
{
    internal class IterationProgressMonitor : ITrainingIterationNotifications
    {
        public void IterationFinished(IterationResult result)
        {
            Console.WriteLine($"Finished iteration, result is {result.Score.RSquared}.");
            Separator();
        }

        public void IterationStarted(Iteration iteration)
        {
            Separator();
            Console.WriteLine($"Iteration started.  Pipeline {iteration.Pipeline}");
        }
        public void IterationFailed(Exception e)
        {
            Console.WriteLine($"Iteration failed: {e}");
        }

        private void Separator()
        {
            Console.WriteLine($"-----------------------------------------");
        }
    }
}
