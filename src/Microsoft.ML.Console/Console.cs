// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Console
    {
        public static int Main(string[] args) {
            var stopwatch = Stopwatch.StartNew();
            var returnVal = Maml.Main(args);
            System.Console.WriteLine($"Time elapsed: {stopwatch.ElapsedMilliseconds}ms");
            return returnVal;
        }
    }
}