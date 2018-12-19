using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public interface ILearnerCatalogItem
    {
        IEnumerable<SweepableParam> GetHyperparamSweepRanges();
        IEstimator<ITransformer> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams);
        string GetLearnerName();
    }

    public class AveragedPerceptronCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> SweepRanges =
            LearnerCatalogUtil.AveragedLinearArgsSweepableParams
                .Concat(LearnerCatalogUtil.OnlineLinearArgsSweepableParams);

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepRanges;
        }

        public IEstimator<ITransformer> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<AveragedPerceptronTrainer.Arguments> argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            return new AveragedPerceptronTrainer(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "AveragedPerceptron";
        }
    }

    public class FastForestCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> SweepRanges = LearnerCatalogUtil.TreeArgsSweepableParams;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepRanges;
        }

        public IEstimator<ITransformer> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<FastForestClassification.Arguments> argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            return new FastForestClassification(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "FastForest";
        }
    }
}
