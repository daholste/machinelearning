// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Core.Data;
using Microsoft.ML.PipelineInference2;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Trainers.SymSgd;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public sealed class DataAndModel<TModel>
    {
        public Var<IDataView> OutData { get; }
        public Var<TModel> Model { get; }

        public DataAndModel(Var<IDataView> outData, Var<TModel> model)
        {
            OutData = outData;
            Model = model;
        }
    }

    public abstract class PipelineNodeBase
    {
        public virtual ParameterSet HyperSweeperParamSet { get; set; }

        protected void PropagateParamSetValues(ParameterSet hyperParams,
            IEnumerable<SweepableParam> sweepParams)
        {
            var spMap = sweepParams.ToDictionary(sp => sp.Name);

            foreach (var hp in hyperParams)
            {
                //Contracts.Assert(spMap.ContainsKey(hp.Name));
                if(spMap.ContainsKey(hp.Name))
                {
                    var sp = spMap[hp.Name];
                    sp.SetUsingValueText(hp.ValueText);
                }
            }
        }
    }

    public sealed class TransformPipelineNode
    {
        public readonly IEstimator<ITransformer> Estimator;

        public TransformPipelineNode(IEstimator<ITransformer> estimator)
        {
            Estimator = estimator;
        }

        public TransformPipelineNode Clone()
        {
            return new TransformPipelineNode(Estimator);
        }
    }

    public sealed class TrainerPipelineNode : PipelineNodeBase
    {
        public readonly string LearnerName;

        public IEnumerable<SweepableParam> SweepParams { get; }

        public TrainerPipelineNode(IEnumerable<SweepableParam> sweepParams = null,
            ParameterSet hyperParameterSet = null, string learnerName = null)
        {
            SweepParams = sweepParams.ToArray();
            HyperSweeperParamSet = hyperParameterSet?.Clone();

            // Make sure sweep params and param set are consistent.
            if (HyperSweeperParamSet != null)
            {
                PropagateParamSetValues(HyperSweeperParamSet, SweepParams);
            }

            LearnerName = learnerName;
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> BuildTrainer(MLContext env)
        {
            ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> trainer;

            if(LearnerName == "AveragedPerceptronBinaryClassifier")
            {
                Action<AveragedPerceptronTrainer.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new AveragedPerceptronTrainer(env, advancedSettings: argsFunc);
            }
            else if (LearnerName == "FastForestBinaryClassifier")
            {
                Action<FastForestClassification.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new FastForestClassification(env, advancedSettings: argsFunc);
            }
            else if (LearnerName == "FastTreeBinaryClassifier")
            {
                Action<FastTreeBinaryClassificationTrainer.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new FastTreeBinaryClassificationTrainer(env, advancedSettings: argsFunc);
            }
            /*else if (LearnerName == "FieldAwareFactorizationMachineBinaryClassifier")
            {
                trainer = new FieldAwareFactorizationMachineTrainer(env, new FieldAwareFactorizationMachineTrainer.Arguments());
            }*/
            else if (LearnerName == "LightGbmBinaryClassifier")
            {
                Action<LightGbmArguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new LightGbmBinaryTrainer(env, advancedSettings: argsFunc);
            }
            else if (LearnerName == "LinearSvmBinaryClassifier")
            {
                var args = new LinearSvm.Arguments();
                if (HyperSweeperParamSet != null)
                {
                    AutoMlUtils.UpdatePropertiesAndFields(args, SweepParams);
                }
                trainer = new LinearSvm(env, args);
            }
            else if (LearnerName == "LogisticRegressionBinaryClassifier")
            {
                Action<LogisticRegression.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new LogisticRegression(env, advancedSettings: argsFunc);
            }
            /*else if (LearnerName == "StochasticDualCoordinateAscentBinaryClassifier")
            {
                trainer = new SdcaBinaryTrainer(env);
            }*/
            else if (LearnerName == "StochasticGradientDescentBinaryClassifier")
            {
                Action<StochasticGradientDescentClassificationTrainer.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new StochasticGradientDescentClassificationTrainer(env, advancedSettings: argsFunc);
            }
            else if (LearnerName == "SymSgdBinaryClassifier")
            {
                Action<SymSgdClassificationTrainer.Arguments> argsFunc = null;
                if (HyperSweeperParamSet != null)
                {
                    argsFunc = (x) => AutoMlUtils.UpdatePropertiesAndFields(x, SweepParams);
                }
                trainer = new SymSgdClassificationTrainer(env, advancedSettings: argsFunc);
            }
            else
            {
                trainer = new AveragedPerceptronTrainer(env);
            }

            return trainer;
        }

        public TrainerPipelineNode Clone() => new TrainerPipelineNode(SweepParams, HyperSweeperParamSet, LearnerName);

        public override string ToString()
        {
            return $"{LearnerName}{{{string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"))}}}";
        }

        public ParameterSet BuildParameterSet()
        {
            return BuildParameterSet(SweepParams);
        }

        private static ParameterSet BuildParameterSet(IEnumerable<SweepableParam> sweepParams)
        {
            var paramValues = new List<IParameterValue>();
            foreach (var sweepParam in sweepParams)
            {
                IParameterValue paramValue = null;
                switch (sweepParam)
                {
                    case SweepableDiscreteParam dp:
                        paramValue = new StringParameterValue(dp.Name, dp.ProcessedValue().ToString());
                        break;
                    case SweepableFloatParam fp:
                        paramValue = new FloatParameterValue(fp.Name, (float)fp.RawValue);
                        break;
                    case SweepableLongParam lp:
                        paramValue = new LongParameterValue(lp.Name, (long)lp.RawValue);
                        break;
                        //default: throw?
                }
                paramValues.Add(paramValue);
            }
            return new ParameterSet(paramValues);
        }
    }
}
