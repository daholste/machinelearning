// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using System.Collections.Concurrent;
using Microsoft.ML.Runtime.Data.IO;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Utilities for various heuristics against text files.
    /// Currently, separator inference and column count detection.
    /// </summary>
    public static class TextFileContents
    {
        public readonly struct ColumnSplitResult
        {
            public readonly bool IsSuccess;
            public readonly string Separator;
            public readonly bool AllowQuote;
            public readonly bool AllowSparse;
            public readonly int ColumnCount;

            public ColumnSplitResult(bool isSuccess, string separator, bool allowQuote, bool allowSparse, int columnCount)
            {
                IsSuccess = isSuccess;
                Separator = separator;
                AllowQuote = allowQuote;
                AllowSparse = allowSparse;
                ColumnCount = columnCount;
            }
        }

        // If the fraction of lines having the same number of columns exceeds this, we consider the column count to be known.
        private const Double UniformColumnCountThreshold = 0.98;

        public static string[] DefaultSeparators = new[] { "tab", ",", ";", " " };

        /// <summary>
        /// Attempt to detect text loader arguments.
        /// The algorithm selects the first 'acceptable' set: the one that recognizes the same number of columns in at
        /// least <see cref="UniformColumnCountThreshold"/> of the sample's lines,
        /// and this number of columns is more than 1.
        /// We sweep on separator, allow sparse and allow quote parameter.
        /// </summary>
    }
}
