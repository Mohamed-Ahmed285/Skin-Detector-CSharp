# Naive Bayes Skin Segmentation Using CSharp

![Language](https://img.shields.io/badge/Language-C%23-blue)
![Framework](https://img.shields.io/badge/Framework-Windows%20Forms-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-success)

A highly optimized Machine Learning application that uses a **Naive Bayes Classifier** to detect and segment human skin pixels in images. This project was developed as part of an Algorithms Warm-up Competition, with a heavy focus on algorithmic efficiency, exact mathematical modeling, and execution speed.

## Key Features & Optimizations

This project calculates the Posterior probabilities using Bayes' Theorem ($P(Skin | Hue)$), but it is heavily optimized to process millions of pixels in milliseconds:

* **O(1) Decision Pre-computation:** Instead of performing floating-point math on every single pixel, the model pre-calculates the True/False classification for all 360 possible Hues before scanning the image. This turns millions of calculations into instant array lookups.
* **Multi-Threading (`Parallel.For`):** Bypasses single-core bottlenecks by processing image rows and batch evaluation datasets simultaneously across all available CPU cores.
* **Zero-Allocation Evaluation:** The evaluation metric pipeline (calculating Precision & Recall) is decoupled from the UI prediction matrix, meaning it calculates True Positives, False Positives, etc., without allocating massive throwaway arrays in RAM.
* **HSV Color Space:** Converts raw RGB pixels to the HSV color cylinder, isolating the **Hue** channel as the primary feature to negate the effects of lighting, shadows, and varying skin tones.

## The Math (Naive Bayes)

The classifier operates on the following rules derived from Bayes' Theorem (dropping the evidence $P(h)$ as it cancels out):

1. **Training:** Calculates the **Prior** probability of skin/non-skin across the dataset, and the **Likelihood** of seeing a specific Hue given a class.
2. **Prediction:** A pixel is classified as "Skin" if and only if:
   * $P(h | Skin) \times P(Skin) > P(h | NonSkin) \times P(NonSkin)$
   * **AND** $P(h | Skin) \times P(Skin) \ge Threshold$ (if a custom threshold is provided).

## Project Structure

* `SkinSegmentation.cs`: The core Machine Learning engine. Contains the heavily optimized `Train`, `Predict`, and `Evaluate` logic.
* `ImageOperations.cs`: Helper library for fast Bitmap I/O, converting RGB to HSV, and strict mask validation.
* `MainForm.cs`: The Windows Forms User Interface for loading datasets, visualizing histograms, and auto-grading against expected model files.

## Usage

1. **Clone the repository** and open the solution in Visual Studio.
2. **Run the Application**.
3. **Train:** Click the `Train` button and select your dataset folder containing colored images and their corresponding black-and-white mask images.
4. **Predict:** Click `Predict` to test the model on a single unseen image. Skin pixels will retain their color, while background pixels will be painted black.
5. **Evaluate:** Click `Evaluate` to run the model against a hidden testing dataset. The UI will output the True Positives, False Positives, True Negatives, False Negatives, along with the final **Precision** and **Recall** percentages.

## Performance

Built to excel in algorithmic execution time metrics, this implementation eliminates redundant memory allocations and mathematical overhead, ensuring an incredibly fast processing pipeline suitable for high-resolution datasets.
