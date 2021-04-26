using System;
using System.Collections.Generic;
using System.Linq;
using System.Data;
using System.IO;

namespace DecisionTree
{
    class Node
    {
        internal string Type;
        private string NodeDesc;
        internal int NodeID;
        internal int NodeDepth;
        internal string splitFeature;
        internal decimal splitPoint;
        internal decimal entropy;
        internal int[] classLabels;
        internal Node Left;
        internal Node Right;
        internal bool isPureLeaf;

        internal DataTable dtSampleData;

        internal Node(int p_nodeId, int p_depth, string p_type, DataTable p_dtSample)
        {
            this.Type = p_type;
            this.NodeID = p_nodeId;
            this.NodeDepth = p_depth;
            this.dtSampleData = p_dtSample.Copy();

            //var allLabels = dtSampleData.AsEnumerable().Select(x => Convert.ToInt32(x["label"])).ToArray();
            var allLabels = Helper.GetDataRow(dtSampleData).Select(x => Convert.ToInt32(x["label"])).ToArray();

            this.entropy = Helper.GetEntropy(allLabels);
            this.isPureLeaf = allLabels.Distinct().Count() == 1 ? true : false;
            this.classLabels = allLabels;

            var dict = allLabels.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            string s = string.Empty;
            foreach(var d in dict )
            {
                s += "[" + d.Key + ":" + d.Value + "], ";
            }

            //this.NodeDesc = $"ID:{p_nodeId}-T:{p_type}-D:{p_depth}-E:{this.entropy}-L:{string.Join(",",allLabels.Distinct())}-[{s}]";
            //Console.WriteLine(this.NodeDesc);

            if (!isPureLeaf && p_depth < 3)
            {
                Tuple<string, decimal, decimal, DataTable, DataTable> InformationGain = Helper.GetInformationGain(dtSampleData);

                this.splitFeature = InformationGain.Item1;
                this.splitPoint = InformationGain.Item2;

                this.Left  = new Node(this.NodeID + 1, this.NodeDepth + 1, $"{this.NodeID} - Left", InformationGain.Item4);
                this.Right = new Node(this.NodeID + 2, this.NodeDepth + 1, $"{this.NodeID} - Right", InformationGain.Item5);
            }
        }
    }

    class Helper
    {
        internal static IEnumerable<DataRow> GetDataRow(DataTable dt)
        {
            List<DataRow> lst = new List<DataRow>();

            foreach(DataRow dr in dt.Rows)
            {
                lst.Add(dr);
            }

            return lst;
        }

        internal static Tuple<DataTable, DataTable> ReadDataFromInput(string path)
        {
            DataTable dtTrainingData = new DataTable("TrainingData");
            DataTable dtTestData = new DataTable("TestData");

            var lines = System.IO.File.ReadAllLines(path);

            //Take the 1st line to get the columns
            var Line1 = lines[0];
            var Line1Split = Line1.Split(' ');

            dtTrainingData.Columns.Add("rowid", typeof(int));
            dtTrainingData.Columns.Add("label", typeof(decimal));

            dtTestData.Columns.Add("rowid", typeof(int));
            dtTestData.Columns.Add("label", typeof(decimal));

            foreach (string split in Line1Split)
            {
                if (split.Contains(':'))
                {
                    var colName = split.Split(':')[0];

                    dtTrainingData.Columns.Add(colName, typeof(decimal));
                    dtTestData.Columns.Add(colName, typeof(decimal));
                }
            }

            int iRow = 0;
            foreach (string line in lines)
            {
                ++iRow;
                DataRow rowTraining = dtTrainingData.NewRow();
                DataRow rowTest = dtTestData.NewRow();

                var ColDataPairs = line.Split(' ');

                int lable = Convert.ToInt32(ColDataPairs[0]);

                for (int i = 0; i < ColDataPairs.Length; i++)
                {
                    string coldatapair = ColDataPairs[i];
                    if (ColDataPairs[i].Contains(':'))
                    {
                        if (lable != -1)
                        {
                            string attName = coldatapair.Split(':')[0];
                            decimal value = Convert.ToDecimal(coldatapair.Split(':')[1]);

                            rowTraining["rowid"] = iRow;
                            rowTraining["label"] = lable;
                            rowTraining[attName] = value;
                        }
                        else
                        {
                            string attName = coldatapair.Split(':')[0];
                            decimal value = Convert.ToDecimal(coldatapair.Split(':')[1]);

                            rowTest["rowid"] = iRow;
                            rowTest["label"] = lable;
                            rowTest[attName] = value;
                        }
                    }
                }

                if (lable != -1)
                    dtTrainingData.Rows.Add(rowTraining);
                else
                    dtTestData.Rows.Add(rowTest);
            }

            return Tuple.Create<DataTable, DataTable>(dtTrainingData, dtTestData);
        }

        internal static Tuple<DataTable, DataTable> LoadDataFromSTDIN()
        {
            DataTable dtTrainingData = new DataTable("TrainingData");
            DataTable dtTestData = new DataTable("TestData");

            string stdin = null;
            if (Console.IsInputRedirected)
            {
                using (StreamReader reader = new StreamReader(Console.OpenStandardInput(), Console.InputEncoding))
                {
                    stdin = reader.ReadToEnd();
                }
            }

            var linesTemp = stdin.Split('\n');

            List<string> lines = new List<string>();
            foreach(string line in linesTemp)
            {
                if (line.Trim().Length > 0)
                    lines.Add(line);
            }

            //Take the 1st line to get the columns
            var Line1 = lines[0];
            var Line1Split = Line1.Split(' ');

            dtTrainingData.Columns.Add("rowid", typeof(int));
            dtTrainingData.Columns.Add("label", typeof(decimal));

            dtTestData.Columns.Add("rowid", typeof(int));
            dtTestData.Columns.Add("label", typeof(decimal));

            foreach (string split in Line1Split)
            {
                if (split.Contains(':'))
                {
                    var colName = split.Split(':')[0];

                    dtTrainingData.Columns.Add(colName, typeof(decimal));
                    dtTestData.Columns.Add(colName, typeof(decimal));
                }
            }

            int iRow = 0;
            foreach (string line in lines)
            {
                ++iRow;
                DataRow rowTraining = dtTrainingData.NewRow();
                DataRow rowTest = dtTestData.NewRow();

                var ColDataPairs = line.Split(' ');

                int lable = Convert.ToInt32(ColDataPairs[0]);

                for (int i = 0; i < ColDataPairs.Length; i++)
                {
                    string coldatapair = ColDataPairs[i];
                    if (ColDataPairs[i].Contains(':'))
                    {
                        if (lable != -1)
                        {
                            string attName = coldatapair.Split(':')[0];
                            decimal value = Convert.ToDecimal(coldatapair.Split(':')[1]);

                            rowTraining["rowid"] = iRow;
                            rowTraining["label"] = lable;
                            rowTraining[attName] = value;
                        }
                        else
                        {
                            string attName = coldatapair.Split(':')[0];
                            decimal value = Convert.ToDecimal(coldatapair.Split(':')[1]);

                            rowTest["rowid"] = iRow;
                            rowTest["label"] = lable;
                            rowTest[attName] = value;
                        }
                    }
                }

                if (lable != -1)
                    dtTrainingData.Rows.Add(rowTraining);
                else
                    dtTestData.Rows.Add(rowTest);
            }

            return Tuple.Create<DataTable, DataTable>(dtTrainingData, dtTestData);
        }

        /// <summary>
        /// columnname, split, informationGain, leftdatatable, rightdatatable
        /// </summary>
        /// <returns></returns>
        internal static Tuple<string, decimal, decimal, DataTable, DataTable> GetInformationGain(DataTable dtSampleData)
        {
            //Get InfoD before split
            //var allLabels = dtSampleData.AsEnumerable().Select(x => Convert.ToInt32(x["label"])).ToArray();
            var allLabels = Helper.GetDataRow(dtSampleData).Select(x => Convert.ToInt32(x["label"])).ToArray();
            decimal infoD_BeforeSplit = GetEntropy(allLabels);
            int TotalSamplesBeforeSplit = dtSampleData.Rows.Count;

            Dictionary<string, decimal[]> columnSplit = GetCoulumnSplit(dtSampleData);

            List<Tuple<string, decimal, decimal, DataTable, DataTable>> InformationGains = new List<Tuple<string, decimal, decimal, DataTable, DataTable>>();
            foreach (var kvp in columnSplit)
            {
                string columnname = kvp.Key;
                decimal[] possibleSplitPoint = kvp.Value;

                foreach(decimal splitvalue in possibleSplitPoint)
                {
                    //Split the samplespace

                    //var RowsLeft = dtSampleData.AsEnumerable().Where(x => Convert.ToDecimal(x[columnname]) <= splitvalue).Select(x=>x);
                    var RowsLeft = Helper.GetDataRow(dtSampleData).Where(x => Convert.ToDecimal(x[columnname]) <= splitvalue).Select(x => x).ToArray();
                    DataTable dtLeft = Helper.CopyToDataTable(RowsLeft);
                    int leftSplitCount = dtLeft.Rows.Count;
                    //int[] leftLabels = dtLeft.AsEnumerable().Select(x => Convert.ToInt32(x["label"])).ToArray();
                    int[] leftLabels = Helper.GetDataRow(dtLeft).Select(x => Convert.ToInt32(x["label"])).ToArray();

                    //var RowsRight = dtSampleData.AsEnumerable().Where(x => Convert.ToDecimal(x[columnname]) > splitvalue).Select(x => x);
                    var RowsRight = Helper.GetDataRow(dtSampleData).Where(x => Convert.ToDecimal(x[columnname]) > splitvalue).Select(x => x).ToArray();
                    DataTable dtRight = Helper.CopyToDataTable(RowsRight);
                    int rightSplitCount = dtRight.Rows.Count;
                    //int[] rightLabels = dtRight.AsEnumerable().Select(x => Convert.ToInt32(x["label"])).ToArray();
                    int[] rightLabels = Helper.GetDataRow(dtRight).Select(x => Convert.ToInt32(x["label"])).ToArray();

                    //Get Info AfterSplit
                    var InfoA = ((decimal)leftSplitCount / (decimal)TotalSamplesBeforeSplit) * GetEntropy(leftLabels) + ((decimal)rightSplitCount / (decimal)TotalSamplesBeforeSplit) * GetEntropy(rightLabels);

                    decimal informationGain = infoD_BeforeSplit - InfoA;

                    InformationGains.Add(Tuple.Create(columnname,splitvalue, informationGain, dtLeft, dtRight));
                }
            }

            //select the Max Gain
            var maxInforGain = InformationGains.OrderByDescending(x => x.Item3).ToArray()[0];
            return maxInforGain;
        }

        internal static decimal GetEntropy(int[] labels)
        {
            //get distinct lablels
            int[] lb = labels.Distinct().ToArray();
            double samples = labels.Count();

            double sumEntropy = 0;
            foreach (int label in lb)
            {
                double labelCount = labels.Where(x => x == label).Count();

                if (labelCount == 0)
                    sumEntropy += 0;
                else
                    sumEntropy += (labelCount / samples) * Math.Log(labelCount / samples, 2.0);
            }

            sumEntropy = sumEntropy * -1;

            return Convert.ToDecimal(sumEntropy);
        }

        private static Dictionary<string, decimal[]> GetCoulumnSplit(DataTable dtSampleData)
        {
            Dictionary<string, decimal[]> ColumnSplitValue = new Dictionary<string, decimal[]>();

            foreach(DataColumn columnname in dtSampleData.Columns)
            {
                if (columnname.ColumnName == "rowid" || columnname.ColumnName == "label") continue;

                //decimal[] SplitPoint = dtSampleData.AsEnumerable().Select(row => Convert.ToDecimal(row[columnname.ColumnName])).OrderBy(x=>x).Distinct().ToArray();
                decimal[] SplitPoint = Helper.GetDataRow(dtSampleData).Select(row => Convert.ToDecimal(row[columnname.ColumnName])).OrderBy(x => x).Distinct().ToArray();

                List<decimal> possibleSplitPoint = new List<decimal>();

                for (int i = 0; i < SplitPoint.Length - 1; i++)
                {
                    var v1 = SplitPoint[i];
                    var v2 = SplitPoint[i + 1];

                    possibleSplitPoint.Add(0.5m * (v1 + v2));
                }

                ColumnSplitValue.Add(columnname.ColumnName, possibleSplitPoint.ToArray());
            }

            return ColumnSplitValue;
        }
    
        internal static Node BuildTree(DataTable sampleSpace)
        {
            Node root = new Node(0, 0, "Root", sampleSpace);
            return root;
        }

        internal static int Predict(Node node, DataRow testRow)
        {
            if (node.isPureLeaf)
            {
                return node.classLabels[0];
            }
            else if (node.NodeDepth == 2)
            {
                //Get the class labels
                var dict = node.classLabels.GroupBy(x => x)
                            .ToDictionary(x => x.Key, x => x.Count());

                //which label has max?
                decimal max = dict.Values.Max(v => v);

                //Is there multiple Max?
                var keys = dict.Where(pair => max.Equals(pair.Value))
                    .Select(pair => pair.Key).ToArray();

                if (keys.Length > 1)
                {
                    return keys.Min(k => k);
                }
                else
                {
                    return keys[0];
                }
            }
            else
            {
                //Get the current node attribute and split value
                string feature = node.splitFeature;
                decimal split = node.splitPoint;

                decimal testValue = Convert.ToDecimal(testRow[feature]);

                if (testValue <= split)
                    return Predict(node.Left, testRow);
                else
                    return Predict(node.Right, testRow);
            }
        }
    
        internal static DataTable CopyToDataTable(DataRow[] dr)
        {
            DataTable dt = dr[0].Table.Clone() ;
            foreach (DataRow d in dr)
            {
                dt.ImportRow(d);
            }
            return dt;
        }
    }


    class Solution
    {
        static void Main(string[] args)
        {
            //Tuple<DataTable, DataTable> inputSet = Helper.ReadDataFromInput(@"TestData/Input21.txt");
            Tuple<DataTable, DataTable> inputSet = Helper.LoadDataFromSTDIN();

            Node tree = Helper.BuildTree(inputSet.Item1);

            //Predict
            foreach(DataRow dr in inputSet.Item2.Rows)
            {
                int predict = Helper.Predict(tree, dr);

                Console.WriteLine(predict);
            }
        }
    }
}
