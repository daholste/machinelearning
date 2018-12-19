using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public class LearnerCatalog
    {
        public static LearnerCatalog Instance = new LearnerCatalog();

        private readonly static IDictionary<MacroUtils.TrainerKinds, IEnumerable<ILearnerCatalogItem>> TasksToLearners =
            new Dictionary<MacroUtils.TrainerKinds, IEnumerable<ILearnerCatalogItem>>()
            {
                { MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                    new ILearnerCatalogItem[] {
                        new AveragedPerceptronCatalogItem(),
                        new FastForestCatalogItem(),
                    } },
            };

        public IEnumerable<ILearnerCatalogItem> GetLearners(MacroUtils.TrainerKinds trainerKind)
        {
            return TasksToLearners[trainerKind];
        }
    }
}
