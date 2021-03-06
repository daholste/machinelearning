// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class InferenceUtils
    {
        public static IDataView Take(this IDataView data, int count)
        {
            //Contracts.CheckValue(data, nameof(data));
            // REVIEW: This should take an env as a parameter, not create one.
            var env = new ConsoleEnvironment(0);
            var take = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeArguments { Count = count }, data);
            return CacheCore(take, env);
        }

        private static IDataView CacheCore(IDataView data, IHostEnvironment env)
        {
            //Contracts.AssertValue(data, "data");
            //Contracts.AssertValue(env, "env");
            return new CacheDataView(env, data, Enumerable.Range(0, data.Schema.ColumnCount).ToArray());
        }

        public static ColumnGroupingInference.GroupingColumn[] InferColumnPurposes(IChannel ch, IHostEnvironment env, TextFileSample sample, TextFileContents.ColumnSplitResult splitResult,
            out bool hasHeader, string colLabelName = null)
        {
           // ch.Info("Detecting column types");
            var typeInferenceResult = ColumnTypeInference.InferTextFileColumnTypes(env, sample,
                new ColumnTypeInference.Arguments
                {
                    ColumnCount = splitResult.ColumnCount,
                    Separator = splitResult.Separator,
                    AllowSparse = splitResult.AllowSparse,
                    AllowQuote = splitResult.AllowQuote,
                });

            hasHeader = true;
            if (!typeInferenceResult.IsSuccess)
            {
                //ch.Error("Couldn't detect column types.");
                return null;
            }

            //ch.Info("Detecting column purposes");
            var typedLoaderArgs = new TextLoader.Arguments
            {
                Column = ColumnTypeInference.GenerateLoaderColumns(typeInferenceResult.Columns),
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote,
                HasHeader = typeInferenceResult.HasHeader
            };
            var typedData = TextLoader.ReadFile(env, typedLoaderArgs, sample);

            var purposeInferenceResult = PurposeInference.InferPurposes(env, typedData,
                Enumerable.Range(0, typedLoaderArgs.Column.Length), new PurposeInference.Arguments(),
                colLabelName: colLabelName);
            //ch.Info("Detecting column grouping and generating column names");

            ColumnGroupingInference.GroupingColumn[] groupingResult = ColumnGroupingInference.InferGroupingAndNames(env, typeInferenceResult.HasHeader,
                typeInferenceResult.Columns, purposeInferenceResult.Columns).Columns;

            return groupingResult;
        }
    }

    // REVIEW: Should this also have the base type (ITrainer<...>)?
    public sealed class PredictorCategory
    {
        public readonly string Name;
        public readonly Type Signature;

        public PredictorCategory(string name, Type sig)
        {
            Name = name;
            Signature = sig;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    public enum ColumnPurpose
    {
        Ignore = 0,
        Name = 1,
        Label = 2,
        NumericFeature = 3,
        CategoricalFeature = 4,
        TextFeature = 5,
        Weight = 6,
        Group = 7,
        ImagePath = 8
    }
}
