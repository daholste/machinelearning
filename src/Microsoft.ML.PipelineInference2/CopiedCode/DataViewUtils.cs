﻿using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference
{
    public static class DataViewUtils
    {
        /// <summary>
        /// Generate a unique temporary column name for the given schema.
        /// Use tag to independently create multiple temporary, unique column
        /// names for a single transform.
        /// </summary>
        public static string GetTemporaryColumnName(this ISchema schema, string tag = null)
        {
            int col;
            if (!string.IsNullOrWhiteSpace(tag) && !schema.TryGetColumnIndex(tag, out col))
                return tag;

            for (int i = 0; ; i++)
            {
                string name = string.IsNullOrWhiteSpace(tag) ?
                    string.Format("temp_{0:000}", i) :
                    string.Format("temp_{0}_{1:000}", tag, i);

                if (!schema.TryGetColumnIndex(name, out col))
                    return name;
            }
        }
    }
}