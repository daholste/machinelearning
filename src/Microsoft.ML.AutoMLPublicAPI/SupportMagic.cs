using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.Runtime.PipelineInference.ColumnGroupingInference;
using static Microsoft.ML.Runtime.PipelineInference.TextFileContents;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public static class StaticSupportMagic
    {
        private static MLContext _mlContext = new MLContext();

        public static TrainContextBase InferTrainContext(this MLContext mlContext, IDataView data)
        {
            return mlContext.Regression;
        }

        public static GroupingColumn[] InferColumns(this MLContext mlContext, string dataFilePath, string label = null)
        {
            var sample = TextFileSample.CreateFromFullFile(mlContext, dataFilePath);
            var splitResult = TextFileContents.TrySplitColumns(mlContext, sample, TextFileContents.DefaultSeparators);
            return InferenceUtils.InferColumnPurposes(null, mlContext, sample, splitResult, out var hasHeader, label);
        }

        public static IDataView InferDataView(this MLContext mlContext, string dataFilePath, GroupingColumn[] columnPurposes)
        {
            var sample = TextFileSample.CreateFromFullFile(mlContext, dataFilePath);
            var splitResult = TextFileContents.TrySplitColumns(mlContext, sample, TextFileContents.DefaultSeparators);
            var args = new TextLoader.Arguments
            {
                Column = ColumnGroupingInference.GenerateLoaderColumns(columnPurposes),
                HasHeader = true,
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote
            };
            var reader = mlContext.Data.TextReader(args);
            return reader.Read(dataFilePath);
        }

        public static PipelineInferenceResult<RegressionPredictionTransformer<LinearRegressionPredictor>> InferPipeline(this AutoMlContext autoMlContext, GroupingColumn[] columns,
            IDataView trainData, IDataView validationData = null, IDataView testData = null,
            int? maxIterations = null)
        {
            // label column transform
            var labelColName = columns.First(c => c.Purpose == ColumnPurpose.Label).SuggestedName;
            var transform = _mlContext.Transforms.CopyColumns(labelColName, "Label");

            var trainer = _mlContext.Regression.Trainers.StochasticDualCoordinateAscent(maxIterations: maxIterations);
            var estimator = transform.Append(trainer);

            var model = estimator.Fit(trainData);
            var transformedOutput = model.Transform(validationData);
            var results = _mlContext.Regression.Evaluate(transformedOutput);
            return new PipelineInferenceResult<RegressionPredictionTransformer<LinearRegressionPredictor>>()
            {
                Estimator = estimator,
                ValidationResult = results,
                TestResult = null,
                TrainedModel = model
            };
        }
    }

    public class IPipelineInferenceResult<T> where T : class, ITransformer
    {
        public EstimatorChain<T> Estimator { get; set; }
        public RegressionEvaluator.Result ValidationResult { get; set; }
        public RegressionEvaluator.Result TestResult { get; set; }
        public ITransformer TrainedModel { get; set; }
    }

    public class PipelineInferenceResult<T> : IPipelineInferenceResult<T> where T : class, ITransformer
    {
    }
}
