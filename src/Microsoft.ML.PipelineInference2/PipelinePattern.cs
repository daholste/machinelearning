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
using Microsoft.ML.Transforms.Normalizers;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    public sealed class PipelinePattern
    {
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
            var ctx = new BinaryClassificationContext(env);
            var metrics = ctx.Evaluate(scoredTestData);
            testMetricValue = metrics.Accuracy;
        }

        public ITransformer TrainTransformer(IDataView trainData, IChannel ch)
        {
            // apply transforms to trian and test data
            IEstimator<ITransformer> estimatorChain = new EstimatorChain<ITransformer>();
            foreach (var transform in Transforms)
            {
                if(transform.PipelineNode.Estimator != null)
                {
                    estimatorChain = estimatorChain.Append(transform.PipelineNode.Estimator);
                }
            }

            // get learner
            var learner = Learner.PipelineNode.BuildTrainer(_env);

            // normalize features, if needed
            if (learner.Info.NeedNormalization)
            {
                var normalizingEstimator = new NormalizingEstimator(_env, DefaultColumnNames.Features);
                estimatorChain = estimatorChain.Append(normalizingEstimator);
            }

            var transformerChain = estimatorChain.Fit(trainData);
            var roleMappedTrainData = new RoleMappedData(trainData, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                RoleMappedSchema.ColumnRole.Feature.Bind(DefaultColumnNames.Features));

            // train learner
            var calibratorFactory = new PlattCalibratorTrainerFactory();
            var caliTrainer = calibratorFactory?.CreateComponent(_env);
            var predictor = TrainUtils.Train(_env, ch, roleMappedTrainData, learner, calibratorFactory, 1000000000);
            return new PipelinePatternTransform(_env, transformerChain, predictor);
        }
    }
}
