using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static SkinSegmentation.ImageOperations;

namespace SkinSegmentation
{
    public struct BayesModel
    {
        public double[] LikelihoodSkin;
        public double[] LikelihoodNonSkin;
        public double PriorSkin;
        public double PriorNonSkin;
    }
    public struct Metrics
    {
        public int TP, TN, FP, FN;
        public double Precision
        {
            get { return 100.0 * (double)TP / (TP + FP); } 
        }

        public double Recall
        {
            get { return 100.0 * (double)TP / (TP + FN); }
        }
    }
    public class SkinSegmentation
    {
        /// <summary>
        /// Train the Bayes model using the given colored images and their corresponding skin mask images
        /// </summary>
        /// <param name="imagePaths">path of the training colored images</param>
        /// <param name="maskPaths">path of the corresponding mask images</param>
        /// <returns>Bayes model (Likelihood & Prior of each class)</returns>
        public static BayesModel Train(string[] imagePaths, string[] maskPaths)
        {
            BayesModel model = new BayesModel();
            int numOfFeatures = 360;
            model.LikelihoodSkin = new double[numOfFeatures];
            model.LikelihoodNonSkin = new double[numOfFeatures];

            long totalSkinPixels = 0;
            long totalNonSkinPixels = 0;
            long[] skinHueCounts = new long[numOfFeatures];
            long[] nonSkinHueCounts = new long[numOfFeatures];

            object lockObj = new object();

          
            Parallel.For(0, imagePaths.Length, i =>
            {
                RGBPixel[,] clrImage = ImageOperations.OpenImage(imagePaths[i]);
                RGBPixel[,] maskImage = ImageOperations.OpenImage(maskPaths[i]);

                int height = ImageOperations.GetHeight(clrImage);
                int width = ImageOperations.GetWidth(clrImage);

                long localSkin = 0;
                long localNonSkin = 0;
                long[] localSkinHue = new long[numOfFeatures];
                long[] localNonSkinHue = new long[numOfFeatures];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        RGBPixel maskPixel = maskImage[y, x];
                        int hue = ImageOperations.ConvertRgbToHsv(clrImage[y, x]).hue;

                        if (hue >= numOfFeatures) hue = numOfFeatures - 1;
                        if (hue < 0) hue = 0;

                        if (maskPixel.red > 0 || maskPixel.green > 0 || maskPixel.blue > 0)
                        {
                            localSkinHue[hue]++;
                            localSkin++;
                        }
                        else
                        {
                            localNonSkinHue[hue]++;
                            localNonSkin++;
                        }
                    }
                }

                // Safely merge this thread's local counts into the global totals
                lock (lockObj)
                {
                    totalSkinPixels += localSkin;
                    totalNonSkinPixels += localNonSkin;
                    for (int h = 0; h < numOfFeatures; h++)
                    {
                        skinHueCounts[h] += localSkinHue[h];
                        nonSkinHueCounts[h] += localNonSkinHue[h];
                    }
                }
            });

            long totalPixels = totalSkinPixels + totalNonSkinPixels;
            if (totalPixels > 0)
            {
                model.PriorSkin = (double)totalSkinPixels / totalPixels;
                model.PriorNonSkin = (double)totalNonSkinPixels / totalPixels;
            }

            for (int h = 0; h < numOfFeatures; h++)
            {
                if (totalSkinPixels > 0)
                    model.LikelihoodSkin[h] = (double)skinHueCounts[h] / totalSkinPixels;
                else
                    model.LikelihoodSkin[h] = 0;

                if (totalNonSkinPixels > 0)
                    model.LikelihoodNonSkin[h] = (double)nonSkinHueCounts[h] / totalNonSkinPixels;
                else
                    model.LikelihoodNonSkin[h] = 0;
            }

            return model;
        }
        /// <summary>
        /// Predict the skin pixels of the given image using the given Bayes Model
        /// </summary>
        /// <param name="imgPath">path of the test image</param>
        /// <param name="model">trained Bayes model</param>
        /// <param name="threshold">threshold of skin posterior</param>
        /// <returns>segmented image</returns>
        public static RGBPixel[,] Predict(string imgPath, BayesModel model, double threshold = 0)
        {
            RGBPixel[,] originalImage = ImageOperations.OpenImage(imgPath);
            int height = ImageOperations.GetHeight(originalImage);
            int width = ImageOperations.GetWidth(originalImage);
            RGBPixel[,] segmentedImage = new RGBPixel[height, width];
            int numOfFeatures = 360;

            bool[] isSkinDecision = new bool[numOfFeatures];
            for (int h = 0; h < numOfFeatures; h++)
            {
                double posteriorSkin = model.LikelihoodSkin[h] * model.PriorSkin;
                double posteriorNonSkin = model.LikelihoodNonSkin[h] * model.PriorNonSkin;

                if (threshold > 0)
                    isSkinDecision[h] = posteriorSkin >= threshold;
                else
                    isSkinDecision[h] = posteriorSkin > posteriorNonSkin;
            }

           
            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    RGBPixel currentPixel = originalImage[y, x];

                    int hue = ImageOperations.ConvertRgbToHsv(currentPixel).hue;
                    if (hue >= numOfFeatures) hue = numOfFeatures - 1;
                    if (hue < 0) hue = 0;

                    if (isSkinDecision[hue])
                    {
                        segmentedImage[y, x] = currentPixel;
                    }
                    else
                    {
                        segmentedImage[y, x].red = 0;
                        segmentedImage[y, x].green = 0;
                        segmentedImage[y, x].blue = 0;
                    }
                }
            });

            return segmentedImage;
        }

        /// <summary>
        /// Evaluate the given Bayes model using the given set of images and their corresponding ground-thruth masks
        /// </summary>
        /// <param name="imagePaths"></param>
        /// <param name="maskPaths"></param>
        /// <param name="threshold"></param>
        /// <returns>Evaluation metrics (Precision & Recall)</returns>
        public static Metrics Evaluate(string[] imagePaths, string[] maskPaths, BayesModel trainedModel, double threshold = 0)
        {
            Metrics metrics = new Metrics();
            metrics.TP = 0; metrics.TN = 0; metrics.FP = 0; metrics.FN = 0;
            int numOfFeatures = 360;

            bool[] isSkinDecision = new bool[numOfFeatures];
            for (int h = 0; h < numOfFeatures; h++)
            {
                double posteriorSkin = trainedModel.LikelihoodSkin[h] * trainedModel.PriorSkin;
                double posteriorNonSkin = trainedModel.LikelihoodNonSkin[h] * trainedModel.PriorNonSkin;
                if (threshold > 0)
                    isSkinDecision[h] = posteriorSkin >= threshold;
                else
                    isSkinDecision[h] = posteriorSkin > posteriorNonSkin;
            }

            object lockObj = new object();

            Parallel.For(0, imagePaths.Length, i =>
            {
                RGBPixel[,] clrImage = ImageOperations.OpenImage(imagePaths[i]);
                RGBPixel[,] actualMask = ImageOperations.OpenImage(maskPaths[i]);

                int height = ImageOperations.GetHeight(clrImage);
                int width = ImageOperations.GetWidth(clrImage);

                int localTP = 0, localTN = 0, localFP = 0, localFN = 0;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int hue = ImageOperations.ConvertRgbToHsv(clrImage[y, x]).hue;
                        if (hue >= numOfFeatures) hue = numOfFeatures - 1;
                        if (hue < 0) hue = 0;

                        bool predictedSkin = isSkinDecision[hue]; // Instant lookup
                        bool actualSkin = actualMask[y, x].red > 0 || actualMask[y, x].green > 0 || actualMask[y, x].blue > 0;

                        if (predictedSkin && actualSkin) localTP++;
                        else if (!predictedSkin && !actualSkin) localTN++;
                        else if (predictedSkin && !actualSkin) localFP++;
                        else if (!predictedSkin && actualSkin) localFN++;
                    }
                }

                
                lock (lockObj)
                {
                    metrics.TP += localTP;
                    metrics.TN += localTN;
                    metrics.FP += localFP;
                    metrics.FN += localFN;
                }
            });

            return metrics;
        }
    }
}
