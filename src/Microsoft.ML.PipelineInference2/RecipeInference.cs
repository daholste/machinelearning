// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.PipelineInference2;

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
                //Contracts.Check(transforms != null, "Transforms cannot be null");
                //Contracts.Check(learners != null, "Learners cannot be null");
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

        public readonly struct InferenceResult
        {
            public readonly SuggestedRecipe[] SuggestedRecipes;
            public InferenceResult(SuggestedRecipe[] suggestedRecipes)
            {
                SuggestedRecipes = suggestedRecipes;
            }
        }

        public abstract class Recipe
        {
            public virtual List<Type> AllowedTransforms() => new List<Type>()
            {
                typeof (TransformInference.Experts.AutoLabel),
                typeof (TransformInference.Experts.GroupIdHashRename),
                typeof (TransformInference.Experts.NameColumnConcatRename),
                typeof (TransformInference.Experts.LabelAdvisory),
                //typeof (TransformInference.Experts.Boolean),
                typeof (TransformInference.Experts.Categorical),
                typeof (TransformInference.Experts.Text),
                typeof (TransformInference.Experts.NumericMissing),
                typeof (TransformInference.Experts.FeaturesColumnConcatRename),
            };

            public virtual List<Type> QualifierTransforms() => AllowedTransforms();

            protected virtual TransformInference.SuggestedTransform[] GetSuggestedTransforms(
                TransformInference.InferenceResult transformInferenceResult, Type predictorType)
            {
                List<Type> allowedTransforms = AllowedTransforms();
                List<Type> qualifierTransforms = QualifierTransforms();

                if (transformInferenceResult.SuggestedTransforms.Any(transform => qualifierTransforms.Contains(transform.ExpertType)))
                {
                    return transformInferenceResult.SuggestedTransforms
                        .Where(transform => allowedTransforms.Contains(transform.ExpertType) || qualifierTransforms.Contains(transform.ExpertType))
                        .ToArray();
                }

                return null;
            }

            public virtual IEnumerable<SuggestedRecipe> Apply(
                TransformInference.InferenceResult transformInferenceResult, Type predictorType, IChannel ch)
            {
                TransformInference.SuggestedTransform[] transforms = GetSuggestedTransforms(
                    transformInferenceResult, predictorType);

                if (transforms?.Length > 0)
                {
                    foreach (var recipe in ApplyCore(predictorType, transforms))
                        yield return recipe;
                }
            }

            protected abstract IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType, TransformInference.SuggestedTransform[] transforms);
        }

        public static TextLoader.Arguments MyAutoMlInferTextLoaderArguments(IHostEnvironment env,
            string dataFile, string labelColName)
        {
            var h = env.Register("InferRecipesFromData", seed: 0, verbose: false);
            using (var ch = h.Start("InferRecipesFromData"))
            {
                var sample = TextFileSample.CreateFromFullFile(h, dataFile);
                var splitResult = TextFileContents.TrySplitColumns(h, sample, TextFileContents.DefaultSeparators);
                var columnPurposes = InferenceUtils.InferColumnPurposes(ch, h, sample, splitResult,
                    out var hasHeader, labelColName);
                return new TextLoader.Arguments
                {
                    Column = ColumnGroupingInference.GenerateLoaderColumns(columnPurposes),
                    HasHeader = true,
                    Separator = splitResult.Separator,
                    AllowSparse = splitResult.AllowSparse,
                    AllowQuoting = splitResult.AllowQuote
                };
            }
        }

        public static List<string> GetLearnerSettingsAndSweepParams(IHostEnvironment env, ComponentCatalog.LoadableClassInfo cl, out string settings)
        {
            List<string> sweepParams = new List<string>();
            var ci = cl.Constructor?.GetParameters();
            if (ci == null)
            {
                settings = "";
                return sweepParams;
            }

            var suggestedSweepsParser = new SuggestedSweepsParser();
            StringBuilder learnerSettings = new StringBuilder();

            foreach (var prop in ci)
            {
                var fieldInfo = prop.ParameterType?.GetFields(BindingFlags.Public | BindingFlags.Instance);

                foreach (var field in fieldInfo)
                {
                    TGUIAttribute[] tgui =
                        field.GetCustomAttributes(typeof(TGUIAttribute), true) as TGUIAttribute[];
                    if (tgui == null)
                        continue;
                    foreach (var attr in tgui)
                    {
                        if (attr.SuggestedSweeps != null)
                        {
                            // Build the learner setting.
                            learnerSettings.Append($" {field.Name}=${field.Name}$");

                            // Build the sweeper.
                            suggestedSweepsParser.TryParseParameter(attr.SuggestedSweeps, field.FieldType, field.Name, out var sweepValues, out var error);
                            sweepParams.Add(sweepValues?.ToStringParameter(env));
                        }
                    }
                }
            }
            settings = learnerSettings.ToString();
            return sweepParams;
        }

        /// <summary>
        /// Given a predictor type returns a set of all permissible learners (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static SuggestedRecipe.SuggestedLearner[] AllowedLearners(IHostEnvironment env, MacroUtils.TrainerKinds trainerKind)
        {
            // for binary classification only
            var learnerNames = new[]
            {
                "AveragedPerceptronBinaryClassifier",
                "FastForestBinaryClassifier",
                "FastTreeBinaryClassifier",
                "LightGbmBinaryClassifier",
                "LinearSvmBinaryClassifier",
                "LogisticRegressionBinaryClassifier",
                "StochasticGradientDescentBinaryClassifier",
                "SymSgdBinaryClassifier",
            };

            var learners = new List<SuggestedRecipe.SuggestedLearner>();
            foreach (var learnerName in learnerNames)
            {
                var sweepParams = AutoMlUtils.GetSweepRangesNewApi(learnerName);
                var learner = new SuggestedRecipe.SuggestedLearner
                {
                    PipelineNode = new TrainerPipelineNode(sweepParams, learnerName: learnerName),
                    LearnerName = learnerName
                };
                learners.Add(learner);
            }

            return learners.ToArray();
        }
    }
}
