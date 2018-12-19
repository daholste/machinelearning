using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;
using static Microsoft.ML.Core.Data.SchemaShape;
using static Microsoft.ML.Core.Data.SchemaShape.Column;
using Microsoft.ML.Runtime;
using static Microsoft.ML.Runtime.PipelineInference.AutoInference;
using static Microsoft.ML.RegressionContext;
using Microsoft.ML.Runtime.PipelineInference;
using System.Linq;
using Microsoft.ML.PipelineInference2;
using static Microsoft.ML.BinaryClassificationContext;

namespace Microsoft.ML.AutoMLPublicAPI2
{
    public class AutoMLResult
    {
        public ITransformer BestModel;
        public IEnumerable<PipelinePattern> AllPipelines;
    }

    public static class MLContextExtensions
    {
        public static AutoMLResult Auto(this BinaryClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            // preprocess train and validation data
            var preprocessorTransform = preprocessor.Fit(trainData);
            trainData = preprocessorTransform.Transform(trainData);
            validationData = preprocessorTransform.Transform(validationData);

            var rocketEngine = new RocketEngine(mlContext, new RocketEngine.Arguments() { });
            var terminator = new IterationTerminator(maxIterations);

            var amls = new AutoMlMlState(mlContext,
                PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Accuracy),
                rocketEngine, terminator, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                   trainData, validationData);
            var pipelineResults = amls.InferPipelines(1, 1, 100);

            // hack: start dummy host & channel
            var host = (mlContext as IHostEnvironment).Register("hi");
            var ch = host.Start("hi");

            var bestPipeline = pipelineResults.First();
            var bestPipelineTransformer = bestPipeline.TrainTransformer(trainData, ch);

            // prepend preprocessors to AutoML model before returning
            var bestModel = preprocessorTransform.Append(bestPipelineTransformer);

            return new AutoMLResult()
            {
                BestModel = bestModel,
                AllPipelines = pipelineResults
            };
        }
    }
}
