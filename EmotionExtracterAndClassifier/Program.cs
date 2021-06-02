using System;
using System.IO;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace EmotionExtracterAndClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            float LeftEyeBrow = 0f;
            float RightEyeBrow = 0f;
            float LeftLip = 0f;
            float RightLip = 0f;
            float LipHeight = 0f;
            float LipWidth = 0f;

            // create and train the model
            var mlContext = new MLContext();
            //IDataView dataView = mlContext.Data.LoadFromTextFile<FaceFeatures>("feature_vectors-training.csv", hasHeader: true, separatorChar: ',');
            //var featureVectorName = "Features";
            //var labelColumnName = "Label";
            //var pipeline =
            //    mlContext.Transforms.Conversion
            //    .MapValueToKey(
            //        inputColumnName: "Emotion",
            //        outputColumnName: labelColumnName)
            //    .Append(mlContext.Transforms.Concatenate(featureVectorName, "LeftEyeBrow", "RightEyeBrow", "LeftLip", "RightLip", "LipHeight", "LipWidth"))
            //    .AppendCacheCheckpoint(mlContext)
            //    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureVectorName))
            //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //var model = pipeline.Fit(dataView);
            //using (var fileStream = new FileStream("emotionModel.zip", FileMode.Create, FileAccess.Write, FileShare.Write)) { mlContext.Model.Save(model, dataView.Schema, fileStream); }




            // load a trained model
            DataViewSchema modelSchema;
            ITransformer model = mlContext.Model.Load("emotionModel.zip", out modelSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<FaceFeatures, EmotionPrediction>(model);



            // Testing Data

            var testDataView = mlContext.Data.LoadFromTextFile<FaceFeatures>("feature_vectors-testing.csv", hasHeader: true, separatorChar: ',');

            var transformedTestData = model.Transform(testDataView);

            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(transformedTestData));

            Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data");
            Console.WriteLine($"* MicroAccuracy: {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"* MacroAccuracy: {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"* LogLoss: {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"* LogLossReduction {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

            // read in an image and extract the features from it
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                Console.WriteLine("Enter the filepath of the image you want to scan");
                string imageFilePath = Console.ReadLine();
                var img = Dlib.LoadImage<RgbPixel>(imageFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // left eyebrow
                    var leftEyebrow1 = (shape.GetPart(18) - shape.GetPart(39)).Length;       // this is 19 - 40
                    var leftEyebrow2 = (shape.GetPart(19) - shape.GetPart(39)).Length;       // this is 20 - 40
                    var leftEyebrow3 = (shape.GetPart(20) - shape.GetPart(39)).Length;       // this is 21 - 40 
                    var leftEyebrow4 = (shape.GetPart(21) - shape.GetPart(39)).Length;       // this is 22 - 40

                    leftEyebrow1 = leftEyebrow1 / leftEyebrow4;
                    leftEyebrow2 = leftEyebrow2 / leftEyebrow4;
                    leftEyebrow3 = leftEyebrow3 / leftEyebrow4;
                    leftEyebrow4 = leftEyebrow4 / leftEyebrow4;

                    var leftEyeBrowSum = leftEyebrow1 + leftEyebrow2 + leftEyebrow3 + leftEyebrow4;
                    LeftEyeBrow = Convert.ToSingle(leftEyeBrowSum);

                    //right eyebrow
                    var rightEyebrow1 = (shape.GetPart(25) - shape.GetPart(42)).Length;     // this is 26 - 43
                    var rightEyebrow2 = (shape.GetPart(24) - shape.GetPart(42)).Length;     // this is 25 - 43
                    var rightEyebrow3 = (shape.GetPart(23) - shape.GetPart(42)).Length;     // this is 24 - 43
                    var rightEyebrow4 = (shape.GetPart(22) - shape.GetPart(42)).Length;     // this is 23 - 43

                    rightEyebrow1 = rightEyebrow1 / rightEyebrow4;
                    rightEyebrow2 = rightEyebrow2 / rightEyebrow4;
                    rightEyebrow3 = rightEyebrow3 / rightEyebrow4;
                    rightEyebrow4 = rightEyebrow4 / rightEyebrow4;

                    var rightEyebrowSum = rightEyebrow1 + rightEyebrow2 + rightEyebrow3 + rightEyebrow4;
                    RightEyeBrow = Convert.ToSingle(rightEyebrowSum);

                    // left lip
                    var lipNormalising = (shape.GetPart(51) - shape.GetPart(33)).Length;    // this is 52 - 34

                    var leftLip1 = (shape.GetPart(48) - shape.GetPart(33)).Length;          // this is 49 - 34
                    var leftLip2 = (shape.GetPart(49) - shape.GetPart(33)).Length;          // this is 50 - 34
                    var leftLip3 = (shape.GetPart(50) - shape.GetPart(33)).Length;          // this is 51 - 34

                    leftLip1 = leftLip1 / lipNormalising;
                    leftLip2 = leftLip2 / lipNormalising;
                    leftLip3 = leftLip3 / lipNormalising;

                    var leftLipSum = leftLip1 + leftLip2 + leftLip3;
                    LeftLip = Convert.ToSingle(leftLipSum);

                    // right lip
                    var rightLip1 = (shape.GetPart(54) - shape.GetPart(33)).Length;         // this is 55 - 34
                    var rightLip2 = (shape.GetPart(53) - shape.GetPart(33)).Length;         // this is 54 - 34
                    var rightLip3 = (shape.GetPart(52) - shape.GetPart(33)).Length;         // this is 53 - 34

                    rightLip1 = rightLip1 / lipNormalising;
                    rightLip2 = rightLip2 / lipNormalising;
                    rightLip3 = rightLip3 / lipNormalising;

                    var rightLipSum = rightLip1 + rightLip2 + rightLip3;
                    RightLip = Convert.ToSingle(rightLipSum);

                    // lip width
                    var lipWidth = (shape.GetPart(48) - shape.GetPart(54)).Length;          // this is 49 - 55
                    lipWidth = lipWidth / lipNormalising;
                    LipWidth = Convert.ToSingle(lipWidth);

                    // lip height
                    var lipHeight = (shape.GetPart(51) - shape.GetPart(57)).Length;         // this is 52 - 58
                    lipHeight = lipHeight / lipNormalising;
                    LipHeight = Convert.ToSingle(lipHeight);

                    Console.WriteLine("\n" + "Left Eyebrow  > " + LeftEyeBrow.ToString());
                    Console.WriteLine("Right Eyebrow > " + RightEyeBrow.ToString());
                    Console.WriteLine("Left Lip      > " + LeftLip.ToString());
                    Console.WriteLine("Right Lip     > " + RightLip.ToString());
                    Console.WriteLine("Lip Height    > " + LipHeight.ToString());
                    Console.WriteLine("Lip Width     > " + LipWidth.ToString() + "\n"); 
                }
                var prediction = predictor.Predict(new FaceFeatures()
                {
                    LeftEyeBrow = LeftEyeBrow,
                    RightEyeBrow = RightEyeBrow,
                    LeftLip = LeftLip,
                    RightLip = RightLip,
                    LipHeight = LipHeight,
                    LipWidth = LipWidth
                });
                // now test the model
                Console.WriteLine($"*** Prediction: {prediction.Emotion} ***");
                Console.WriteLine($"*** Scores: {string.Join(" ", prediction.Scores)} ***");
            }
        }

        public class FaceFeatures
        {
            [LoadColumn(0)]
            public float LeftEyeBrow { get; set; }

            [LoadColumn(1)]
            public float RightEyeBrow { get; set; }

            [LoadColumn(2)]
            public float LeftLip { get; set; }

            [LoadColumn(3)]
            public float RightLip { get; set; }

            [LoadColumn(4)]
            public float LipHeight { get; set; }

            [LoadColumn(5)]
            public float LipWidth { get; set; }

            [LoadColumn(6)]
            public string Emotion { get; set; }
        }

        public class EmotionPrediction
        {
            [ColumnName("PredictedLabel")]
            public string Emotion { get; set; }

            [ColumnName("Score")]
            public float[] Scores { get; set; }
        }
    }
}
