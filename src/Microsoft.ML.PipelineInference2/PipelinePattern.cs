// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.PipelineInference;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.PipelineInference2;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    public sealed class PipelinePattern
    {
        /// <summary>
        /// Class for encapsulating the information returned in the output IDataView for a pipeline
        /// that has been run through the TrainTest macro.
        /// </summary>
        public sealed class PipelineResultRow
        {
            public string GraphJson { get; }
            ///<summary>
            /// The metric value of the test dataset result (always needed).
            ///</summary>
            public double MetricValue { get; }
            ///<summary>
            /// The metric value of the training dataset result (not always used or set).
            ///</summary>
            public double TrainingMetricValue { get; }
            public string PipelineId { get; }
            public string FirstInput { get; }
            public string PredictorModel { get; }

            public PipelineResultRow()
            { }

            public PipelineResultRow(string graphJson, double metricValue,
                string pipelineId, double trainingMetricValue, string firstInput,
                string predictorModel)
            {
                GraphJson = graphJson;
                MetricValue = metricValue;
                PipelineId = pipelineId;
                TrainingMetricValue = trainingMetricValue;
                FirstInput = firstInput;
                PredictorModel = predictorModel;
            }
        }

        private readonly IHostEnvironment _env;
        public readonly TransformInference.SuggestedTransform[] Transforms;
        public readonly RecipeInference.SuggestedRecipe.SuggestedLearner Learner;
        public PipelineSweeperRunSummary PerformanceSummary { get; set; }
        public string LoaderSettings { get; set; }
        public Guid UniqueId { get; }

        public PipelinePattern(TransformInference.SuggestedTransform[] transforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            string loaderSettings, IHostEnvironment env, PipelineSweeperRunSummary summary = null)
        {
            // Make sure internal pipeline nodes and sweep params are cloned, not shared.
            // Cloning the transforms and learner rather than assigning outright
            // ensures that this will be the case. Doing this here allows us to not
            // worry about changing hyperparameter values in candidate pipelines
            // possibly overwritting other pipelines.
            Transforms = transforms.Select(t => t.Clone()).ToArray();
            Learner = learner.Clone();
            LoaderSettings = loaderSettings;
            _env = env;
            PerformanceSummary = summary;
            UniqueId = Guid.NewGuid();
        }

        /// <summary>
        /// This method will return some indentifying string for the pipeline,
        /// based on transforms, learner, and (eventually) hyperparameters.
        /// </summary>
        public override string ToString() => $"{Learner}+{string.Join("+", Transforms.Select(t => t.ToString()))}";

        /// <summary>
        /// Runs a train-test experiment on the current pipeline
        /// </summary>
        public void RunTrainTestExperiment(IDataView trainData, IDataView testData,
            SupportedMetric metric, MacroUtils.TrainerKinds trainerKind, IHostEnvironment env,
            IChannel ch, out double testMetricValue)
        {
            var pipelineTransformer = TrainTransformer(trainData, ch);
            var scoredTestData = pipelineTransformer.Transform(testData);
            var ctx = new RegressionContext(env);
            var metrics = ctx.Evaluate(scoredTestData);
            testMetricValue = metrics.RSquared;
        }

        public ITransformer TrainTransformer(IDataView trainData, IChannel ch)
        {
            // apply transforms to trian and test data
            var estimatorChain = new EstimatorChain<ITransformer>();
            foreach (var transform in Transforms)
            {
                if(transform.PipelineNode.Estimator != null)
                {
                    estimatorChain = estimatorChain.Append(transform.PipelineNode.Estimator);
                }
            }

            var transformerChain = estimatorChain.Fit(trainData);
            trainData = transformerChain.Transform(trainData);

            // get learner
            var learner = Learner.PipelineNode.BuildTrainer(_env);

            // add normalizers
            //TrainUtils.AddNormalizerIfNeeded(env, ch, learner, ref trainData, "Features", Data.NormalizeOption.Auto);
            //roleMappedTestData = ApplyTransformUtils.ApplyAllTransformsToData(env, scoredTestData, scoredTestData);

            var roleMappedTrainData = new RoleMappedData(trainData, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                RoleMappedSchema.ColumnRole.Feature.Bind(DefaultColumnNames.Features));

            // train learner
            var calibratorFactory = new PlattCalibratorTrainerFactory();
            var caliTrainer = calibratorFactory?.CreateComponent(_env);
            var predictor = TrainUtils.Train(_env, ch, roleMappedTrainData, learner, calibratorFactory, 1000000000);
            return new PipelinePatternTransform(_env, transformerChain, predictor);
        }

        public static PipelineResultRow[] ExtractResults(IHostEnvironment env, IDataView data,
            string graphColName, string metricColName, string idColName, string trainingMetricColName,
            string firstInputColName, string predictorModelColName)
        {
            var results = new List<PipelineResultRow>();
            var schema = data.Schema;
            if (!schema.TryGetColumnIndex(graphColName, out var graphCol))
                throw new Exception($"Column name {graphColName} not found");
                //throw env.ExceptParam(nameof(graphColName), $"Column name {graphColName} not found");
            if (!schema.TryGetColumnIndex(metricColName, out var metricCol))
                new Exception($"Column name {metricColName} not found");
                //throw env.ExceptParam(nameof(metricColName), $"Column name {metricColName} not found");
            if (!schema.TryGetColumnIndex(trainingMetricColName, out var trainingMetricCol))
                new Exception($"Column name {trainingMetricColName} not found");
                //throw env.ExceptParam(nameof(trainingMetricColName), $"Column name {trainingMetricColName} not found");
            if (!schema.TryGetColumnIndex(idColName, out var pipelineIdCol))
                new Exception($"Column name {idColName} not found");
                //throw env.ExceptParam(nameof(idColName), $"Column name {idColName} not found");
            if (!schema.TryGetColumnIndex(firstInputColName, out var firstInputCol))
                new Exception($"Column name {firstInputColName} not found");
                //throw env.ExceptParam(nameof(firstInputColName), $"Column name {firstInputColName} not found");
            if (!schema.TryGetColumnIndex(predictorModelColName, out var predictorModelCol))
                new Exception($"Column name {predictorModelColName} not found");
                //throw env.ExceptParam(nameof(predictorModelColName), $"Column name {predictorModelColName} not found");

            using (var cursor = data.GetRowCursor(col => true))
            {
                var getter1 = cursor.GetGetter<double>(metricCol);
                var getter2 = cursor.GetGetter<ReadOnlyMemory<char>>(graphCol);
                var getter3 = cursor.GetGetter<ReadOnlyMemory<char>>(pipelineIdCol);
                var getter4 = cursor.GetGetter<double>(trainingMetricCol);
                var getter5 = cursor.GetGetter<ReadOnlyMemory<char>>(firstInputCol);
                var getter6 = cursor.GetGetter<ReadOnlyMemory<char>>(predictorModelCol);
                double metricValue = 0;
                double trainingMetricValue = 0;
                ReadOnlyMemory<char> graphJson = default;
                ReadOnlyMemory<char> pipelineId = default;
                ReadOnlyMemory<char> firstInput = default;
                ReadOnlyMemory<char> predictorModel = default;

                while (cursor.MoveNext())
                {
                    getter1(ref metricValue);
                    getter2(ref graphJson);
                    getter3(ref pipelineId);
                    getter4(ref trainingMetricValue);
                    getter5(ref firstInput);
                    getter6(ref predictorModel);

                    results.Add(new PipelineResultRow(graphJson.ToString(),
                        metricValue, pipelineId.ToString(), trainingMetricValue,
                        firstInput.ToString(), predictorModel.ToString()));
                }
            }

            return results.ToArray();
        }
    }
}
