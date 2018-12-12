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

        public static IDataView ApplyTransformSet(IDataView data, TransformInference.SuggestedTransform[] transforms)
        {
            foreach(var transform in transforms)
            {
                data = transform.PipelineNode.Estimator.Fit(data).Transform(data);
            }
            return data;
        }
        
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

            var fields = learnerInputType.GetFields(bindingFlags);
            var properties = learnerInputType.GetProperties(bindingFlags);
            var members = fields.Cast<MemberInfo>().Concat(properties);

            var fieldValueGetters = fields.Select(f => new MyFieldInfo(f));
            var propertyValueGetters = properties.Select(p => new MyPropertyInfo(p));
            var memberValueGetters = fieldValueGetters.Cast<IMemberInfo>().Concat(propertyValueGetters);

            var instance = Activator.CreateInstance(learnerInputType);

            for (var i = 0; i < members.Count(); i++)
            {
                var member = members.ElementAt(i);
                var memberValueGetter = memberValueGetters.ElementAt(i);

                var sweepableAttrs = member.GetCustomAttributes(typeof(TlcModule.SweepableParamAttribute), true);
                if(!sweepableAttrs.Any())
                {
                    continue;
                }
                var sweepableAttr = sweepableAttrs.First() as TlcModule.SweepableParamAttribute;
                sweepableAttr.Name = sweepableAttr.Name ?? member.Name;

                var memberValue = memberValueGetter.GetValue(instance);
                if (memberValue != null)
                {
                    sweepableAttr.SetUsingValueText(memberValue.ToString());
                }

                paramSet.Add(sweepableAttr);
            }

            return paramSet.ToArray();
        }

        public interface IMemberInfo
        {
            object GetValue(object obj);
        }

        public class MyPropertyInfo : IMemberInfo
        {
            private PropertyInfo _info;

            public MyPropertyInfo(PropertyInfo info)
            {
                _info = info;
            }

            public object GetValue(object obj)
            {
                return _info.GetValue(obj);
            }
        }

        public class MyFieldInfo : IMemberInfo
        {
            private FieldInfo _info;

            public MyFieldInfo(FieldInfo info)
            {
                _info = info;
            }

            public object GetValue(object obj)
            {
                return _info.GetValue(obj);
            }
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
        public static bool UpdateProperties(object entryPointObj, IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
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
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
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

        public static bool UpdatePropertiesAndFields(object entryPointObj, IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
        {
            var result = UpdateProperties(entryPointObj, sweepParams);
            result &= UpdateFields(entryPointObj, sweepParams);
            return result;
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>
        public static bool UpdateFields(object entryPointObj, IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
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
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
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

        public static bool CheckEntryPointStateMatchesParamValues(object entryPointObj,
            IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
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

        public static IRunResult ConvertToRunResult(RecipeInference.SuggestedRecipe.SuggestedLearner learner, PipelineSweeperRunSummary rs, bool isMetricMaximizing)
        {
            return new RunResult(learner.PipelineNode.BuildParameterSet(), rs.MetricValue, isMetricMaximizing);
        }

        public static IRunResult[] ConvertToRunResults(PipelinePattern[] history, bool isMetricMaximizing)
        {
            return history.Select(h => ConvertToRunResult(h.Learner, h.PerformanceSummary, isMetricMaximizing)).ToArray();
        }

        /// <summary>
        /// Method to convert set of sweepable hyperparameters into <see cref="IComponentFactory"/> instances used
        /// by the current smart hyperparameter sweepers.
        /// </summary>
        public static IComponentFactory<IValueGenerator>[] ConvertToComponentFactories(IEnumerable<TlcModule.SweepableParamAttribute> hps)
        {
            var results = new IComponentFactory<IValueGenerator>[hps.Count()];

            for (int i = 0; i < hps.Count(); i++)
            {
                switch (hps.ElementAt(i))
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
    }
}
