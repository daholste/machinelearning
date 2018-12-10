// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Trainers.SymSgd;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class AutoMlUtils
    {
        public static double ExtractValueFromIdv(IHostEnvironment env, IDataView result, string columnName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(result, nameof(result));
            env.CheckNonEmpty(columnName, nameof(columnName));

            double outputValue = 0;
            var schema = result.Schema;
            if (!schema.TryGetColumnIndex(columnName, out var metricCol))
                throw env.ExceptParam(nameof(columnName), $"Schema does not contain column: {columnName}");

            using (var cursor = result.GetRowCursor(col => col == metricCol))
            {
                var getter = cursor.GetGetter<double>(metricCol);
                bool moved = cursor.MoveNext();
                env.Check(moved, "Expected an IDataView with a single row. Results dataset has no rows to extract.");
                getter(ref outputValue);
                env.Check(!cursor.MoveNext(), "Expected an IDataView with a single row. Results dataset has too many rows.");
            }

            return outputValue;
        }

        /// <summary>
        /// Using the dependencyMapping and included transforms, computes which subset of columns in dataSample
        /// will be present in the final transformed dataset when only the transforms present are applied.
        /// </summary>
        private static int[] GetExcludedColumnIndices(TransformInference.SuggestedTransform[] includedTransforms, IDataView dataSample,
            AutoInference.DependencyMap dependencyMapping)
        {
            List<int> includedColumnIndices = new List<int>();

            // For every column, see if either present in initial dataset, or
            // produced by a transform used in current pipeline.
            for (int columnIndex = 0; columnIndex < dataSample.Schema.ColumnCount; columnIndex++)
            {
                // Create ColumnInfo object for indexing dictionary
                var colInfo = new AutoInference.ColumnInfo
                {
                    Name = dataSample.Schema.GetColumnName(columnIndex),
                    ItemType = dataSample.Schema.GetColumnType(columnIndex).ItemType,
                    IsHidden = dataSample.Schema.IsHidden(columnIndex)
                };

                // Exclude all hidden and non-numeric columns
                if (colInfo.IsHidden || !colInfo.ItemType.IsNumber)
                    continue;

                foreach (var level in dependencyMapping.Keys.Reverse())
                {
                    var levelResponsibilities = dependencyMapping[level];

                    if (!levelResponsibilities.ContainsKey(colInfo))
                        continue;

                    // Include any numeric column present in initial dataset. Does not need
                    // any transforms applied to be present in final dataset.
                    if (level == 0 && colInfo.ItemType.IsNumber && levelResponsibilities[colInfo].Count == 0)
                    {
                        includedColumnIndices.Add(columnIndex);
                        break;
                    }

                    // If column could not have been produced by transforms at this level, move down to the next level.
                    if (levelResponsibilities[colInfo].Count == 0)
                        continue;

                    // Check if could have been produced by any transform in this pipeline
                    if (levelResponsibilities[colInfo].Any(t => includedTransforms.Contains(t)))
                        includedColumnIndices.Add(columnIndex);
                }
            }

            // Exclude all columns not discovered by our inclusion process
            return Enumerable.Range(0, dataSample.Schema.ColumnCount).Except(includedColumnIndices).ToArray();
        }

        public static bool AtomicGroupPresent(long bitmask, int atomicGroupId) => (bitmask & (1 << atomicGroupId)) > 0;

        public static long TransformsToBitmask(TransformInference.SuggestedTransform[] transforms) =>
            transforms.Aggregate(0, (current, t) => current | 1 << t.AtomicGroupId);

        /// <summary>
        /// Gets a final transform to concatenate all numeric columns into a "Features" vector column.
        /// Note: May return empty set if Features column already present and is only relevant numeric column.
        /// (In other words, if there would be nothing for that concatenate transform to do.)
        /// </summary>
        private static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(IHostEnvironment env,
            IDataView dataSample, int[] excludedColumnIndices, int level, int atomicIdOffset, RoleMappedData dataRoles)
        {
            var finalArgs = new TransformInference.Arguments
            {
                EstimatedSampleFraction = 1.0,
                ExcludeFeaturesConcatTransforms = false,
                ExcludedColumnIndices = excludedColumnIndices
            };

            var featuresConcatTransforms = TransformInference.InferConcatNumericFeatures(env, dataSample, finalArgs, dataRoles);

            for (int i = 0; i < featuresConcatTransforms.Length; i++)
            {
                featuresConcatTransforms[i].RoutingStructure.Level = level;
                featuresConcatTransforms[i].AtomicGroupId += atomicIdOffset;
            }

            return featuresConcatTransforms.ToArray();
        }

        /// <summary>
        /// Exposed version of the method.
        /// </summary>
        public static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(IHostEnvironment env, IDataView data,
            AutoInference.DependencyMap dependencyMapping, TransformInference.SuggestedTransform[] selectedTransforms,
            TransformInference.SuggestedTransform[] allTransforms, RoleMappedData dataRoles)
        {
            int level = 1;
            int atomicGroupLimit = 0;
            if (allTransforms.Length != 0)
            {
                level = allTransforms.Max(t => t.RoutingStructure.Level) + 1;
                atomicGroupLimit = allTransforms.Max(t => t.AtomicGroupId) + 1;
            }
            var excludedColumnIndices = GetExcludedColumnIndices(selectedTransforms, data, dependencyMapping);
            return GetFinalFeatureConcat(env, data, excludedColumnIndices, level, atomicGroupLimit, dataRoles);
        }

        /// <summary>
        /// Creates a dictionary mapping column names to the transforms which could have produced them.
        /// </summary>
        public static AutoInference.LevelDependencyMap ComputeColumnResponsibilities(IDataView transformedData,
            TransformInference.SuggestedTransform[] appliedTransforms)
        {
            var mapping = new AutoInference.LevelDependencyMap();
            for (int i = 0; i < transformedData.Schema.ColumnCount; i++)
            {
                if (transformedData.Schema.IsHidden(i))
                    continue;
                var colInfo = new AutoInference.ColumnInfo
                {
                    IsHidden = false,
                    ItemType = transformedData.Schema.GetColumnType(i).ItemType,
                    Name = transformedData.Schema.GetColumnName(i)
                };
                mapping.Add(colInfo, appliedTransforms.Where(t =>
                    t.RoutingStructure.ColumnsProduced.Any(o => o.Name == colInfo.Name &&
                    o.IsNumeric == transformedData.Schema.GetColumnType(i).ItemType.IsNumber)).ToList());
            }
            return mapping;
        }

        public static TlcModule.SweepableParamAttribute[] GetSweepRangesNewApi(string learnerName)
        {
            Type argsType = null;
            if (learnerName == "AveragedPerceptronBinaryClassifier")
            {
                argsType = typeof(AveragedPerceptronTrainer.Arguments);
            }
            else if (learnerName == "FastForestBinaryClassifier")
            {
                argsType = typeof(FastForestClassification.Arguments);
            }
            else if (learnerName == "FastTreeBinaryClassifier")
            {
                argsType = typeof(FastTreeBinaryClassificationTrainer.Arguments);
            }
            else if (learnerName == "LightGbmBinaryClassifier")
            {
                argsType = typeof(LightGbmArguments);
            }
            else if (learnerName == "LinearSvmBinaryClassifier")
            {
                argsType = typeof(LinearSvm.Arguments);
            }
            else if (learnerName == "LogisticRegressionBinaryClassifier")
            {
                argsType = typeof(LogisticRegression.Arguments);
            }
            else if (learnerName == "StochasticDualCoordinateAscentBinaryClassifier")
            {
                argsType = typeof(SdcaBinaryTrainer.Arguments);
            }
            else if (learnerName == "StochasticGradientDescentBinaryClassifier")
            {
                argsType = typeof(StochasticGradientDescentClassificationTrainer.Arguments);
            }
            else if (learnerName == "SymSgdBinaryClassifier")
            {
                argsType = typeof(SymSgdClassificationTrainer.Arguments);
            }
            else
            {
                argsType = typeof(AveragedPerceptronTrainer.Arguments);
            }
            return GetSweepRanges(argsType);
        }

        public static TlcModule.SweepableParamAttribute[] GetSweepRanges(Type learnerInputType)
        {
            var paramSet = new List<TlcModule.SweepableParamAttribute>();

            var bindingFlags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public;
            var members = learnerInputType.GetFields(bindingFlags).Cast<MemberInfo>()
                .Concat(learnerInputType.GetProperties(bindingFlags));

            foreach (var member in members)
            {
                if (member.GetCustomAttributes(typeof(TlcModule.SweepableLongParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableLongParamAttribute lpAttr)
                {
                    lpAttr.Name = lpAttr.Name ?? member.Name;
                    paramSet.Add(lpAttr);
                }

                if (member.GetCustomAttributes(typeof(TlcModule.SweepableFloatParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableFloatParamAttribute fpAttr)
                {
                    fpAttr.Name = fpAttr.Name ?? member.Name;
                    paramSet.Add(fpAttr);
                }

                if (member.GetCustomAttributes(typeof(TlcModule.SweepableDiscreteParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableDiscreteParamAttribute dpAttr)
                {
                    dpAttr.Name = dpAttr.Name ?? member.Name;
                    paramSet.Add(dpAttr);
                }
            }

            return paramSet.ToArray();
        }

        public static IValueGenerator ToIValueGenerator(TlcModule.SweepableParamAttribute attr)
        {
            if (attr is TlcModule.SweepableLongParamAttribute sweepableLongParamAttr)
            {
                var args = new LongParamArguments
                {
                    Min = sweepableLongParamAttr.Min,
                    Max = sweepableLongParamAttr.Max,
                    LogBase = sweepableLongParamAttr.IsLogScale,
                    Name = sweepableLongParamAttr.Name,
                    StepSize = sweepableLongParamAttr.StepSize
                };
                if (sweepableLongParamAttr.NumSteps != null)
                    args.NumSteps = (int)sweepableLongParamAttr.NumSteps;
                return new LongValueGenerator(args);
            }

            if (attr is TlcModule.SweepableFloatParamAttribute sweepableFloatParamAttr)
            {
                var args = new FloatParamArguments
                {
                    Min = sweepableFloatParamAttr.Min,
                    Max = sweepableFloatParamAttr.Max,
                    LogBase = sweepableFloatParamAttr.IsLogScale,
                    Name = sweepableFloatParamAttr.Name,
                    StepSize = sweepableFloatParamAttr.StepSize
                };
                if (sweepableFloatParamAttr.NumSteps != null)
                    args.NumSteps = (int)sweepableFloatParamAttr.NumSteps;
                return new FloatValueGenerator(args);
            }

            if (attr is TlcModule.SweepableDiscreteParamAttribute sweepableDiscreteParamAttr)
            {
                var args = new DiscreteParamArguments
                {
                    Name = sweepableDiscreteParamAttr.Name,
                    Values = sweepableDiscreteParamAttr.Options.Select(o => o.ToString()).ToArray()
                };
                return new DiscreteValueGenerator(args);
            }

            throw new Exception($"Sweeping only supported for Discrete, Long, and Float parameter types. Unrecognized type {attr.GetType()}");
        }

        private static void SetValue(PropertyInfo pi, IComparable value, object entryPointObj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                pi.SetValue(entryPointObj, value);
            else if (propertyType == typeof(double) && value is float)
                pi.SetValue(entryPointObj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                pi.SetValue(entryPointObj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                pi.SetValue(entryPointObj, Convert.ToInt64(value));
        }

        private static void SetValue(FieldInfo fi, IComparable value, object entryPointObj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                fi.SetValue(entryPointObj, value);
            else if (propertyType == typeof(double) && value is float)
                fi.SetValue(entryPointObj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                fi.SetValue(entryPointObj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                fi.SetValue(entryPointObj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>
        public static bool UpdateProperties(object entryPointObj, TlcModule.SweepableParamAttribute[] sweepParams)
        {
            bool result = true;
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    var pi = entryPointObj.GetType().GetProperty(param.Name);
                    if (pi is null || param.RawValue == null)
                        continue;
                    var propType = Nullable.GetUnderlyingType(pi.PropertyType) ?? pi.PropertyType;

                    if (param is TlcModule.SweepableDiscreteParamAttribute dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(pi.PropertyType) != null)
                                pi.SetValue(entryPointObj, null);
                            else if (pi.PropertyType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = pi.PropertyType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(pi.PropertyType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    pi.SetValue(entryPointObj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(pi, (IComparable)dp.Options[optIndex], entryPointObj, propType);
                    }
                    else
                        SetValue(pi, param.RawValue, entryPointObj, propType);
                }
                catch (Exception)
                {
                    // Could not update param
                    result = false;
                }
            }

            // Make sure all changes were saved.
            return result && CheckEntryPointStateMatchesParamValues(entryPointObj, sweepParams);
        }

        public static bool UpdatePropertiesAndFields(object entryPointObj, TlcModule.SweepableParamAttribute[] sweepParams)
        {
            var result = UpdateProperties(entryPointObj, sweepParams);
            result &= UpdateFields(entryPointObj, sweepParams);
            return result;
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>
        public static bool UpdateFields(object entryPointObj, TlcModule.SweepableParamAttribute[] sweepParams)
        {
            bool result = true;
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    var fi = entryPointObj.GetType().GetField(param.Name);
                    if (fi is null || param.RawValue == null)
                        continue;
                    var propType = Nullable.GetUnderlyingType(fi.FieldType) ?? fi.FieldType;

                    if (param is TlcModule.SweepableDiscreteParamAttribute dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(fi.FieldType) != null)
                                fi.SetValue(entryPointObj, null);
                            else if (fi.FieldType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = fi.FieldType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(fi.FieldType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    fi.SetValue(entryPointObj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(fi, (IComparable)dp.Options[optIndex], entryPointObj, propType);
                    }
                    else
                        SetValue(fi, param.RawValue, entryPointObj, propType);
                }
                catch (Exception)
                {
                    // Could not update param
                    result = false;
                }
            }

            // Make sure all changes were saved.
            return result && CheckEntryPointStateMatchesParamValues(entryPointObj, sweepParams);
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>
        public static void PopulateSweepableParams(RecipeInference.SuggestedRecipe.SuggestedLearner learner)
        {
            foreach (var param in learner.PipelineNode.SweepParams)
            {
                if (param is TlcModule.SweepableDiscreteParamAttribute dp)
                {
                    var learnerVal = learner.PipelineNode.GetPropertyValueByName(dp.Name, (IComparable)dp.Options[0]);
                    param.RawValue = dp.IndexOf(learnerVal);
                }
                else if (param is TlcModule.SweepableFloatParamAttribute fp)
                    param.RawValue = learner.PipelineNode.GetPropertyValueByName(fp.Name, 0f);
                else if (param is TlcModule.SweepableLongParamAttribute lp)
                    param.RawValue = learner.PipelineNode.GetPropertyValueByName(lp.Name, 0L);
            }
        }

        public static bool CheckEntryPointStateMatchesParamValues(object entryPointObj,
            TlcModule.SweepableParamAttribute[] sweepParams)
        {
            foreach (var param in sweepParams)
            {
                var pi = entryPointObj.GetType().GetProperty(param.Name);
                if (pi is null)
                    continue;

                // Make sure the value matches
                var epVal = pi.GetValue(entryPointObj);
                if (param.RawValue != null
                    && (!param.ProcessedValue().ToString().ToLower().Contains("auto") || epVal != null)
                    && !epVal.Equals(param.ProcessedValue()))
                    return false;
            }
            return true;
        }

        public static double ProcessWeight(double weight, double maxWeight, bool isMaximizingMetric) =>
            isMaximizingMetric ? weight : maxWeight - weight;

        public static long IncludeMandatoryTransforms(List<TransformInference.SuggestedTransform> availableTransforms) =>
            TransformsToBitmask(GetMandatoryTransforms(availableTransforms.ToArray()));

        public static TransformInference.SuggestedTransform[] GetMandatoryTransforms(
            TransformInference.SuggestedTransform[] availableTransforms) =>
            availableTransforms.Where(t => t.AlwaysInclude).ToArray();

        private static ParameterSet ConvertToParameterSet(TlcModule.SweepableParamAttribute[] hps,
            RecipeInference.SuggestedRecipe.SuggestedLearner learner)
        {
            if (learner.PipelineNode.HyperSweeperParamSet != null)
                return learner.PipelineNode.HyperSweeperParamSet;

            var paramValues = new IParameterValue[hps.Length];

            if (hps.Any(p => p.RawValue == null))
                PopulateSweepableParams(learner);

            for (int i = 0; i < hps.Length; i++)
            {
                Contracts.CheckValue(hps[i].RawValue, nameof(TlcModule.SweepableParamAttribute.RawValue));

                switch (hps[i])
                {
                    case TlcModule.SweepableDiscreteParamAttribute dp:
                        var learnerVal =
                            learner.PipelineNode.GetPropertyValueByName(dp.Name, (IComparable)dp.Options[0]);
                        var optionIndex = (int)(dp.RawValue ?? dp.IndexOf(learnerVal));
                        paramValues[i] = new StringParameterValue(dp.Name, dp.Options[optionIndex].ToString());
                        break;
                    case TlcModule.SweepableFloatParamAttribute fp:
                        paramValues[i] =
                            new FloatParameterValue(fp.Name,
                                (float)(fp.RawValue ?? learner.PipelineNode.GetPropertyValueByName(fp.Name, 0f)));
                        break;
                    case TlcModule.SweepableLongParamAttribute lp:
                        paramValues[i] =
                            new LongParameterValue(lp.Name,
                                (long)(lp.RawValue ?? learner.PipelineNode.GetPropertyValueByName(lp.Name, 0L)));
                        break;
                }
            }

            learner.PipelineNode.HyperSweeperParamSet = new ParameterSet(paramValues);
            return learner.PipelineNode.HyperSweeperParamSet;
        }

        public static IRunResult ConvertToRunResult(RecipeInference.SuggestedRecipe.SuggestedLearner learner, PipelineSweeperRunSummary rs, bool isMetricMaximizing)
        {
            return new RunResult(ConvertToParameterSet(learner.PipelineNode.SweepParams, learner), rs.MetricValue, isMetricMaximizing);
        }

        public static IRunResult[] ConvertToRunResults(PipelinePattern[] history, bool isMetricMaximizing)
        {
            return history.Select(h => ConvertToRunResult(h.Learner, h.PerformanceSummary, isMetricMaximizing)).ToArray();
        }

        /// <summary>
        /// Method to convert set of sweepable hyperparameters into <see cref="IComponentFactory"/> instances used
        /// by the current smart hyperparameter sweepers.
        /// </summary>
        public static IComponentFactory<IValueGenerator>[] ConvertToComponentFactories(TlcModule.SweepableParamAttribute[] hps)
        {
            var results = new IComponentFactory<IValueGenerator>[hps.Length];

            for (int i = 0; i < hps.Length; i++)
            {
                switch (hps[i])
                {
                    case TlcModule.SweepableDiscreteParamAttribute dp:
                        results[i] = ComponentFactoryUtils.CreateFromFunction(env =>
                        {
                            var dpArgs = new DiscreteParamArguments()
                            {
                                Name = dp.Name,
                                Values = dp.Options.Select(o => o.ToString()).ToArray()
                            };
                            return new DiscreteValueGenerator(dpArgs);
                        });
                        break;

                    case TlcModule.SweepableFloatParamAttribute fp:
                        results[i] = ComponentFactoryUtils.CreateFromFunction(env =>
                        {
                            var fpArgs = new FloatParamArguments()
                            {
                                Name = fp.Name,
                                Min = fp.Min,
                                Max = fp.Max,
                                LogBase = fp.IsLogScale,
                            };
                            if (fp.NumSteps.HasValue)
                            {
                                fpArgs.NumSteps = fp.NumSteps.Value;
                            }
                            if (fp.StepSize.HasValue)
                            {
                                fpArgs.StepSize = fp.StepSize.Value;
                            }
                            return new FloatValueGenerator(fpArgs);
                        });
                        break;

                    case TlcModule.SweepableLongParamAttribute lp:
                        results[i] = ComponentFactoryUtils.CreateFromFunction(env =>
                        {
                            var lpArgs = new LongParamArguments()
                            {
                                Name = lp.Name,
                                Min = lp.Min,
                                Max = lp.Max,
                                LogBase = lp.IsLogScale
                            };
                            if (lp.NumSteps.HasValue)
                            {
                                lpArgs.NumSteps = lp.NumSteps.Value;
                            }
                            if (lp.StepSize.HasValue)
                            {
                                lpArgs.StepSize = lp.StepSize.Value;
                            }
                            return new LongValueGenerator(lpArgs);
                        });
                        break;
                }
            }
            return results;
        }

        public static string GenerateOverallTrainingMetricVarName(Guid id) => $"Var_Training_OM_{id:N}";
    }
}
