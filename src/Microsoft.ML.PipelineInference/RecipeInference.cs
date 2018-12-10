// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class RecipeInference
    {
        public readonly struct SuggestedRecipe
        {
            public readonly string Description;
            public readonly TransformInference.SuggestedTransform[] Transforms;
            public struct SuggestedLearner
            {
                public ComponentCatalog.LoadableClassInfo LoadableClassInfo;
                public string Settings;
                public TrainerPipelineNode PipelineNode;
                public string LearnerName;

                public SuggestedLearner Clone()
                {
                    return new SuggestedLearner
                    {
                        LoadableClassInfo = LoadableClassInfo,
                        Settings = Settings,
                        PipelineNode = PipelineNode.Clone(),
                        LearnerName = LearnerName
                    };
                }

                public override string ToString() => PipelineNode.ToString();
            }

            public readonly SuggestedLearner[] Learners;
            public readonly int PreferenceIndex;

            public SuggestedRecipe(string description,
                TransformInference.SuggestedTransform[] transforms,
                SuggestedLearner[] learners,
                int preferenceIndex = -1)
            {
                Contracts.Check(transforms != null, "Transforms cannot be null");
                Contracts.Check(learners != null, "Learners cannot be null");
                Description = description;
                Transforms = transforms;
                Learners = FillLearnerNames(learners);
                PreferenceIndex = preferenceIndex;
            }

            private static SuggestedLearner[] FillLearnerNames(SuggestedLearner[] learners)
            {
                for (int i = 0; i < learners.Length; i++)
                    learners[i].LearnerName = learners[i].LoadableClassInfo.LoadNames[0];
                return learners;
            }

            public override string ToString() => Description;
        }

        /// <summary>
        /// Given a predictor type returns a set of all permissible learners (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static SuggestedRecipe.SuggestedLearner[] AllowedLearners(IHostEnvironment env, MacroUtils.TrainerKinds trainerKind)
        {
            //not all learners advertised in the API are available in CORE.
            var catalog = env.ComponentCatalog;
            var availableLearnersList = catalog.AllEntryPoints().Where(
                x => x.InputKinds?.FirstOrDefault(i => i == typeof(CommonInputs.ITrainerInput)) != null);

            var learners = new List<SuggestedRecipe.SuggestedLearner>();
            var type = typeof(CommonInputs.ITrainerInput);
            var trainerTypes = typeof(Experiment).Assembly.GetTypes()
                .Where(p => type.IsAssignableFrom(p) &&
                    MacroUtils.IsTrainerOfKind(p, trainerKind));
            trainerTypes = trainerTypes.Where(y => !typeof(GeneralizedAdditiveModelBinaryClassifier).Equals(y));
            trainerTypes = trainerTypes.Where(y => !typeof(GeneralizedAdditiveModelRegressor).Equals(y));
            trainerTypes = trainerTypes.Where(y => !typeof(EnsembleBinaryClassifier).Equals(y));
            trainerTypes = trainerTypes.Where(y => !typeof(OnlineGradientDescentRegressor).Equals(y));
            trainerTypes = trainerTypes.Where(y => !typeof(FastTreeTweedieRegressor).Equals(y));

            foreach (var tt in trainerTypes)
            {
                var sweepParams = AutoMlUtils.GetSweepRanges(tt);
                var sweepParams2 = AutoMlUtils.GetSweepRangesNewApi(tt.Name);
                var epInputObj = (CommonInputs.ITrainerInput)tt.GetConstructor(Type.EmptyTypes)?.Invoke(new object[] { });
                var sl = new SuggestedRecipe.SuggestedLearner
                {
                    PipelineNode = new TrainerPipelineNode(epInputObj, sweepParams, learnerName: tt.Name, sweepParamsNewApi: sweepParams2),
                    LearnerName = tt.Name
                };

                if (sl.PipelineNode != null && availableLearnersList.FirstOrDefault(l => l.Name.Equals(sl.PipelineNode.GetEpName())) != null)
                    learners.Add(sl);
            }

            return learners.ToArray();
        }
    }
}
