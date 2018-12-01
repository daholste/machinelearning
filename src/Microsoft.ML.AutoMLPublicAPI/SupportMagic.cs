using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.RegressionContext;
using static Microsoft.ML.Runtime.PipelineInference.ColumnGroupingInference;

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

        public static AutoMlEngine Auto(this RegressionTrainers trainers, Dictionary<string, ColumnPurpose> columnPurposes,
            int? maxIterations = null)
        {
            var mlContext = new MLContext();
            var labelColName = columnPurposes.First(c => c.Value == ColumnPurpose.Label).Key;

            var labelCol = new SchemaShape.Column(labelColName, SchemaShape.Column.VectorKind.Scalar, PrimitiveType.FromKind(DataKind.R4), false);
            var featureCol = new SchemaShape.Column("TripDistance", SchemaShape.Column.VectorKind.Scalar, PrimitiveType.FromKind(DataKind.R4), false);

            return new AutoMlEngine(mlContext, featureCol, labelCol, columnPurposes, maxIterations);
        }
    }

    public class AutoMlEngine : TrainerEstimatorBase<AutoMlISingleFeaturePredictionTransformer, AutoMlPredictor>
    {
        private Dictionary<string, ColumnPurpose> _columnPurposes;
        private int? _maxIterations;

        public AutoMlEngine(IHostEnvironment host,
            SchemaShape.Column feature,
            SchemaShape.Column label,
            Dictionary<string, ColumnPurpose> columnPurposes, int? maxIterations = null,
            SchemaShape.Column weight = null) : base(BuildHost(host), feature, label, weight)
        {
            _columnPurposes = columnPurposes;
            _maxIterations = maxIterations;
        }

        private static IHost BuildHost(IHostEnvironment env)
        {
            return env.Register("hi");
        }

        public override TrainerInfo Info => new TrainerInfo();

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override AutoMlISingleFeaturePredictionTransformer MakeTransformer(AutoMlPredictor predictor, Schema trainSchema)
        {
            var mlContext = new MLContext();

            var labelColName = _columnPurposes.First(c => c.Value == ColumnPurpose.Label).Key;
            var transform1 = mlContext.Transforms.CopyColumns(labelColName, "Label");
            var transform2 = transform1.Append(mlContext.Transforms.Concatenate("Features",
                new string[] { "TripDistance" }));
            //new string[] { "VendorId", "RateCode", "TripTime", "TripDistance", "PaymentType"}));

            var trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(maxIterations: _maxIterations);
            var estimator = transform2.Append(trainer);
            return new AutoMlISingleFeaturePredictionTransformer(estimator);
        }

        protected override AutoMlPredictor TrainModelCore(TrainContext trainContext)
        {
            return new AutoMlPredictor(trainContext);
        }

        protected override RoleMappedData MakeRoles(IDataView data)
        {
            var labelColName = _columnPurposes.First(c => c.Value == ColumnPurpose.Label).Key;
            return new RoleMappedData(data, label: labelColName, feature: null, weight: WeightColumn?.Name);
        }
    }

    public class AutoMlPredictor : IPredictorProducing<float>
    {
        public PredictionKind PredictionKind => throw new NotImplementedException();
        public TrainContext TrainContext { get; private set; }

        public AutoMlPredictor(TrainContext trainContext)
        {
            TrainContext = trainContext;
        }
    }

    public class AutoMlISingleFeaturePredictionTransformer : ISingleFeaturePredictionTransformer<AutoMlPredictor>
    {
        private IEstimator<ITransformer> _estimator;
        private Schema _schema;

        public AutoMlISingleFeaturePredictionTransformer(IEstimator<ITransformer> estimator)
        {
            _estimator = estimator;
        }

        public string FeatureColumn => throw new NotImplementedException();

        public ColumnType FeatureColumnType => throw new NotImplementedException();

        public AutoMlPredictor Model => throw new NotImplementedException();

        public bool IsRowToRowMapper => throw new NotImplementedException();

        public Schema GetOutputSchema(Schema inputSchema)
        {
            return _schema;
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            throw new NotImplementedException();
        }

        public IDataView Transform(IDataView input)
        {
            var model = _estimator.Fit(input);
            var output = model.Transform(input);
            _schema = output.Schema;
            return output;
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

    public enum ColumnPurpose
    {
        Label,
        Categorical,
        Numerical
    }
}
