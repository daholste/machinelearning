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
    public static class AutoMlExtension
    {
        public static AutoMlBinaryClassificationEstimator Auto(this BinaryClassificationTrainers trainers,
            int maxIterations = 10, IDataView validationData = null)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            return new AutoMlBinaryClassificationEstimator(mlContext, maxIterations, validationData);
        }
    }

    public class AutoMlBinaryClassificationEstimator : IEstimator<ITransformer>
    {
        private readonly IHostEnvironment _env;
        private readonly int _maxIterations;

        public IDataView ValidationData { get; set; }

        public AutoMlBinaryClassificationEstimator(IHostEnvironment env, int maxIterations = 10, IDataView validationData = null)
        {
            _env = env;
            _maxIterations = maxIterations;
            ValidationData = validationData;
        }

        public ITransformer Fit(IDataView trainData)
        {
            var rocketEngine = new RocketEngine(_env, new RocketEngine.Arguments() { });
            var terminator = new IterationTerminator(_maxIterations);

            var amls = new AutoMlMlState(_env,
                PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.RSquared),
                rocketEngine, terminator, MacroUtils.TrainerKinds.SignatureRegressorTrainer,
                   trainData, ValidationData);
            var bestPipelines = amls.InferPipelines(1, 1, 100);
            var bestPipeline = bestPipelines.First();

            // hack: start dummy host & channel
            var host = _env.Register("hi");
            var ch = host.Start("hi");

            var transformer = bestPipeline.TrainTransformer(trainData, ch);
            return transformer;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // get list of existing columns
            var cols = inputSchema.Columns.ToList();

            // add score column
            var scoreCol = new SchemaShape.Column("PredictedLabel ", VectorKind.Scalar, BoolType.Instance, false);
            cols.Add(scoreCol);

            return new SchemaShape(cols);
        }
    }

    public enum ColumnPurpose
    {
        Label,
        Categorical,
        Numerical
    }
}
