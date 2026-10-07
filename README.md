# CS4243 Lab 1 implementation report

Author : Caio ARTUS 

## Gabor Implementation

When correlated with an image, a Gabor filter extracts the patterns in the image that match it. In this part we implement a Gabor filter bank, combining Gabor filters of various orientations, frequencies, and phases. The energy is then computed from the response of these filters, and pooled in order to increase the stability of the responses.

We implement the `make_gabor_bank` function which takes in a configuration object indicating the phases, orientations and frequencies of the Gabor filters to be added to the bank. The filters are then created in a deterministic order, iterating over frequencies first, then orientations, then phases. Each kernel has its mean subtracted and is normalised to unit Euclidean norm. The function returns these filters as 2D numpy arrays, where each cell is the value of the 2D Gabor filter at those coordinates. The metadata for each filter is also kept and returned by the function.

Next we want to be able to correlate these filters with an image and compute the response energy maps for each filter. To do this we implement the `gabor_energy_maps` function. This function correlates each map in a given Gabor filter bank with the image, computes the energy (either the squared response or the absolute response depending on the `energy` parameter), and then mean pools these energies with a `pool_size` × `pool_size` box filter, without changing the image height or width.

### Question 1 : How do frequency, orientation, phases, and pooling size change the response maps? 

**Frequency impact**
A higher frequency means the filter responds to stripes that are closer together. By combining filters of different frequencies we can detect stripes with different spacings. Figure 1 shows this on a synthetic example. Notice how a Gabor filter of a certain frequency only matches lines with the same frequency.
<p align="center"><img src="student_files/report/synth_freq.png" width="80%" alt="Synthetic example showing the effect of frequency in the Gabor filter."><br><em><strong>Figure 1.</strong> Synthetic example showing the effect of frequency in the Gabor filter.</em></p>

**Orientation Impact**
Orientation changes which direction the pattern must have to cause a high response. For example an orientation of 0 means the filter responds highly to completely vertical lines, whereas an orientation of $\frac{\pi}{2}$ means the filter responds to completely horizontal lines. The synthetic example in Figure 2 shows this in action: the Gabor filters only match the patterns with the correct orientation.

<p align="center"><img src="student_files/report/synth_orientation.png" width="80%" alt="Synthetic example showing the effect of orientation in the Gabor filter."><br><em><strong>Figure 2.</strong> Synthetic example showing the effect of orientation in the Gabor filter.</em></p>

**Phase impact**

Phase shifts the sinusoidal stripes inside the fixed Gaussian envelope without altering their spacing or orientation. Since our carrier is a sine, when the phase offset is $0$ (or $\pi$) the filter is **antisymmetric** (odd-symmetric) and acts as an edge detector, responding most strongly to sharp light-to-dark transitions. Conversely, when the phase is $\frac{\pi}{2}$ (or $\frac{3\pi}{2}$), the carrier becomes a cosine and the filter is **symmetric** (even-symmetric) and acts as a ridge/stripe detector, responding strongest to line structures.

The synthetic example in Figure 3 illustrates this behaviour as both symmetric and antisymmetric filters process a uniform band and a sharp step edge. The antisymmetric filter produces dual peak responses aligned with the boundaries, whereas the symmetric filter produces a single peak centered directly over the band itself.

<p align="center"><img src="student_files/report/synth_phase.png" width="55%" alt="Synthetic example showing the effect of phase in a Gabor filter (symmetric vs antisymmetric) on a uniform band and a step edge."><br><em><strong>Figure 3.</strong> Synthetic example showing the effect of phase in a Gabor filter (symmetric vs antisymmetric) on a uniform band and a step edge.</em></p>

**Pooling size impact**

Pooling size affects the size of the regions we are considering. With a small pooling size, we look at more local responses. This means that even if we have a pattern that matches our filter (for example the stripes in the wood), if the pooling size is smaller than a period then there will still be peaks and troughs, leading to an unstable response. What we want is for responses in homogeneous regions (same texture) to be constant. A larger pooling size allows us to smooth over these small variations, in particular if we choose a pooling size that is bigger than the wavelength of the filter. Then peaks and troughs due to the filter cancel out in the average. The risk however, is to smooth over important details that are smaller than the pooling size.

For example, consider a wood training image, which has no defects. We expect a relatively smooth response, as the pattern repeats with perhaps some mild variations due to the way the frequency of the stripes changes in the wood. Figure 4 shows the result of varying the pooling size from 3 to 21 when applying a vertical Gabor filter that matches the stripes of the wood. With pooling size 3 we notice stripes in the responses of this vertical filter due to the filter's peaks and troughs. With a larger pooling size these cancel out leaving only the pattern response.


<p align="center"><img src="student_files/report/pooling_stability.png" width="80%" alt="Increasing the pooling size decreases instability."><br><em><strong>Figure 4.</strong> Increasing the pooling size decreases instability.</em></p>

However, Figure 5 shows what happens if we have a small detail we would like to preserve. In this case we choose an image of wood with a defect (holes) and apply a vertical Gabor filter. When pooling becomes too large the peaks in response caused by the smaller holes get smoothed over and confounded with the background texture response.


<p align="center"><img src="student_files/report/pooling_defect.png" width="80%" alt="Too large a pooling size can cause small details to get lost. Here the small holes near the bottom are smoothed over and lost in the larger pooling sizes."><br><em><strong>Figure 5.</strong> Too large a pooling size can cause small details to get lost. Here the small holes near the bottom are smoothed over and lost in the larger pooling sizes.</em></p>

## Edge Implementation

In order to find the edges we implement Canny edge detection. This involves the 4 steps below:

1. Gaussian Smoothing
2. Gradients magnitudes and orientations computation
3. Non maximum suppression 
4. Hysteresis thresholding

The main pipeline is executed in the `detect_edges` function which runs all of these steps. For Gaussian smoothing and gradient magnitude and orientation computation, we used the supplied `gaussian_smoothing`, and `sobel_gradients` functions respectively.

After gradient computation, the gradient magnitude forms a bell-shaped profile across each edge, causing thick edges. Non maximum suppression aims to only keep the pixel with the highest gradient magnitude value on the edge in order to get thinner edges. To do this we look at gradient magnitude across the edge (along the gradient) and only keep pixels with greater values than their neighbors. 

Non maximum suppression can be implemented in two different ways: 

- **Interpolated NMS** uses bilinear interpolation to approximate the gradient magnitude values at the exact coordinates of the gradient vector on either side of the pixel of interest. 

- **Nearest-direction NMS** simply considers the nearest whole pixel value on both sides of the pixel of interest along the gradient direction. 

In `nms_interpolated` we implement the interpolated NMS. This function relies on `bilinear_sample`, which allows us to sample at any coordinates on our image by interpolating with the surrounding pixels. In effect, the formula is:

$$
I(x,y) = (1-w_x)(1-w_y)\,I[y_0,x_0] + w_x(1-w_y)\,I[y_0,x_1] + (1-w_x)w_y\,I[y_1,x_0] + w_xw_y\,I[y_1,x_1]
$$

where $x_0=\lfloor x\rfloor$, $x_1=\min(x_0+1,\,W-1)$, $y_0=\lfloor y\rfloor$, $y_1=\min(y_0+1,\,H-1)$, $w_x=x-x_0$, $w_y=y-y_0$, and $(x,y)$ is first clipped to the image bounds.

Using this function, `nms_interpolated` samples values in the forward direction ($(x_{f},y_{f}) = (x + \cos\theta,\; y + \sin\theta)$) and in the backward direction ($(x_{b},y_{b}) = (x - \cos\theta,\; y - \sin\theta)$) so each pixel can be compared to its neighbours along the gradient direction. Only pixels that are at least as large as both neighbours are kept; the rest, as well as a one-pixel border, are set to 0 in the output.

For hysteresis we need to define low and high thresholds. Instead of being set by hand, `adaptive_thresholds` automatically determines these thresholds. The high threshold is determined either by selecting a percentile of the gradient magnitudes (passed in the configuration as a hyperparameter), or via a robust standard deviation estimate (median absolute deviation, MAD) given by: 

$$\text{high} = \text{median}(x) + 3.7065 \cdot \text{MAD}(x)$$

where $3.7065 = 2.5 \times 1.4826$. Both modes only use the strictly positive NMS values. The low threshold is then calculated by taking a fraction of the high threshold. This fraction is passed in the configuration and is a hyperparameter.

The `hysteresis` function then performs hysteresis in order to only keep weak edges which are connected to strong edges. This function can use either 4 or 8 connectivity. In 4 connectivity we only consider directly adjacent pixels (top, bottom, left and right) as being connected to the current pixel, whereas with 8 connectivity we also consider diagonal pixels (top left, top right, bottom left, bottom right) as being connected. This means that 8 connectivity keeps more edges than 4 connectivity as the criterion for weak edges to be connected to a strong edge is looser. To perform hysteresis we initialise a queue with all the strong pixel coordinates. Then, while the queue is not empty, we pop a coordinate from it and find any weak pixels connected to this pixel. If a pixel is found to be connected and has not yet been considered we add it to the queue so it can also be treated and connected to its neighbors.


### Question 2 : Interpolated NMS versus nearest-direction NMS, 4 vs 8 connectivity

**Interpolated NMS versus nearest-direction NMS**

Nearest direction NMS may fail to accurately detect slanted edges or curves because it forces the gradient direction to snap to one of four discrete directions (0°, 45°, 90°, 135°), causing abrupt discrete changes that often break up continuous edges. Interpolated NMS, on the other hand, ensures continuous transitions by following the exact curve, providing a much more stable and accurate estimated maximum along the edge.

As an example, Figure 6 shows what happens on a grid image where there are lots of curved edges as each cell of the grid looks like a kind of ellipse. We notice that both methods tend to agree on the vertical edges but the nearest-direction approximation either breaks up or completely ignores curved edges, whereas the interpolated version does a good job at keeping them. 

<p align="center"><img src="student_files/report/q2_nms_grid.png" width="55%" alt="Interpolated vs nearest-direction non-maximum suppression on a grid image."><br><em><strong>Figure 6.</strong> Interpolated vs nearest-direction non-maximum suppression on a grid image.</em></p>


If we look at the histogram of the magnitudes of the positive responses (Figure 7), we see again that the interpolated version keeps many more of the pixels. Figure 7 also shows another agreement map where we see that the edges missing from the nearest-direction approximation are mostly slanted edges. Interestingly, some edges are also present in the nearest-direction approximation but not in the interpolated version. These are mostly also artefacts of the discrete approximation. 

<p align="center"><img src="student_files/report/q2_nms_grid_hist.png" width="75%" alt="Interpolated vs nearest-direction non-maximum suppression: histogram of positive response magnitudes and agreement between the two methods."><br><em><strong>Figure 7.</strong> Interpolated vs nearest-direction non-maximum suppression: histogram of positive response magnitudes and agreement between the two methods.</em></p>


**4 vs 8 connectivity**

Connectivity has an impact on what edges are kept in the final edge map. 4 connectivity tends to keep fewer edges, losing or breaking up slanted edges especially, as they tend to be connected via diagonal pixels. Figure 8 shows this on an image of wood where there are many slanted edges. We notice that the 4-connectivity fails to capture a lot of these slanted edges and focuses only on the mostly vertical ones.

<p align="center"><img src="student_files/report/q2_connectivity.png" width="85%" alt="Effect of connectivity on the edge map for a wood image."><br><em><strong>Figure 8.</strong> Effect of connectivity on the edge map for a wood image.</em></p>



## Features and Representations

In this section we aim to build features from our images in order to be able to train a classification model to determine texture class. To do this we want to transform our images into feature vectors that we can feed into a classification algorithm, as feeding raw pixel values to these simple algorithms is unlikely to yield good results. Unlike in deep learning, where the features are learned automatically via backpropagation through a convolutional neural network, we build these features manually using the tools built in previous sections.


We implement the `extract_local_features` function, which constructs a dense feature map ($H \times W \times D$) by combining up to four feature families according to a `FeatureConfig` instance:

1. **Input Normalization & Preprocessing:** Converts the input image to a float32 array normalized to $[0, 1]$, ensures 3-channel RGB alignment, and generates a grayscale image for downstream filter operations.
2. **Colour Features:** Raw $R, G, B$ color channels extracted directly from the image.
3. **Gabor Features:** Spatial energy maps computed across a multi-frequency and multi-orientation Gabor filter bank with spatial pooling.
4. **Gradient Energy Features:** Mean-pooled squared gradient magnitudes computed using a local box filter of size `density_size`.
5. **Edge Features:** Local density of connected (hysteresis) edges computed via spatial box filtering of size `density_size`, along with $N$ orientation-binned edge density maps covering $[0, \pi)$.

Each feature family can be toggled via the configuration object. Enabled features are concatenated along the feature dimension to form an $H \times W \times D$ map alongside a list of $D$ channel name strings. Optionally, per-channel z-score standardization (zero mean, unit variance) is applied spatially across the feature map.

The helper `feature_family_indices` maps each channel name back to its family (colour, Gabor, gradient or edge). We use it in the error analysis of Task A to sum feature contributions per family.


To collapse the dense local feature map into a single global representation for downstream classification or retrieval, `global_pool` aggregates spatial details across the entire image into summary statistics. By flattening the spatial dimensions into pixel-wise feature vectors and computing statistics like mean, standard deviation, and key percentiles (e.g., 90th percentile) per channel, it creates a fixed-size vector that captures both the average presence and the extreme activations of features across the image. This makes the final descriptor invariant to image size and spatial translation of features, as global pooling aggregates statistics across all spatial locations regardless of where a feature occurs. However, the descriptor is not invariant to rotation. Rotating the image alters edge orientations and directional textures, activating different Gabor filter channels and edge orientation bins. For example, rotating a wood grain texture changes its dominant edge angles, resulting in a distinct descriptor vector.


Thus, our final feature vector is a concatenation of summary statistics for each feature channel: 

$$
\mathbf{x} = \big[\;
\underbrace{\text{mean}(\text{colour}), \dots, \text{mean}(\text{edge})}_{\text{mean of each channel}},\;
\underbrace{\text{std}(\text{colour}), \dots, \text{std}(\text{edge})}_{\text{std of each channel}},\;
\underbrace{\text{p90}(\text{colour}), \dots, \text{p90}(\text{edge})}_{\text{90th percentile of each channel}}
\;\big]
$$


### Question 3: Why is standardising the features important?

Figure 9 shows the scale disparities across feature families: while Colour (RGB) features have both a high mean and high variance because of uncentered pixel intensities over a wide dynamic range, Edge features keep a low overall mean with high variance due to sparse boundary spikes, and Gabor features stay low in both mean and variance from zero-mean filtering and spatial pooling. Left unstandardized, these scale differences allow high-magnitude or high-variance features to dominate distance metrics and gradient updates, while causing regularization to penalize model weights unevenly. Standardizing all features to zero mean and unit variance ensures every feature family contributes equitably, making the optimization landscape much better suited for downstream classification.

<p align="center"><img src="student_files/report/q3_scale.png" width="70%" alt="Mean and standard deviation of the features, by feature family."><br><em><strong>Figure 9.</strong> Mean and standard deviation of the features, by feature family.</em></p>

Note that this figure was produced with the default Gabor bank from `config.py` (3 frequencies: 0.08, 0.16 and 0.28, times 4 orientations, so 12 Gabor channels), whereas the pipeline used everywhere else has 4 Gabor channels. The conclusion about the scale differences between feature families is the same.


### Classification using Logistic Regression

Once we have constructed our features, we can now train a classification model to determine the texture class of each image. 

First we load our train (20 samples, 4 per material), validation (100 samples, 10 good and 10 defective per material) and test data (99 samples, 10 good and 10 defective per material, except wood that only has 9 examples with no defects).
We can visualise the first two principal components of our features in order to get some intuition on the linear separability of the data. Figure 10 shows a PCA on all of the data. This is only for analysis: our model is only fitted on training data, and the test data stays held out.

**Note:** The "wood" class only has 9 good images in the test split (the even/odd split of the 19 good wood images leaves 9 in test), instead of 10. This shouldn't change the analysis drastically but is important to note.

<p align="center"><img src="student_files/report/material_PCA.png" width="50%" alt="PCA projection of all the material features onto the first two principal components."><br><em><strong>Figure 10.</strong> PCA projection of all the material features onto the first two principal components.</em></p>


To fit our data, we use the supplied `fit_material_classifier` function which builds a pipeline that scales the data using `StandardScaler` and fits a multiclass logistic regression (with regularisation $C = 5$). 

As metrics we use accuracy and macro F1 (detailed later).


### Configuration

The table below records the configuration values and random seeds used for the reported results (everything else keeps the defaults in `config.py`).

| Component | Setting | Value |
|:---|:---|:---|
| Images | Size | resized to 64×64 |
| Gabor | Frequencies | 0.10, 0.22 |
| Gabor | Orientations | 0, π/2 |
| Gabor | Phases | one sine phase (0) |
| Gabor | σ / kernel size / pool size | 3 / 11 / 7 |
| Gabor | Energy | squared |
| Edges | Gaussian σ | 1.2 |
| Edges | Threshold | "percentile" mode, 90th percentile, low ratio 0.4 |
| Edges | Connectivity | 8 |
| Edges | Density box size | 9 |
| Edges | Orientation bins | 4 |
| Pooling | Statistics | mean, std, p90 (13 channels × 3 = 39 dimensions) |
| Material classifier | Model | StandardScaler + LogisticRegression, C = 5, random_state 7 |
| Attribute classifier (Task C) | Model | one-vs-rest, C = 2, random_state 7 |
| Normality model | Image score | 99th percentile of the interior |
| Normality model | Score pooling / ignored border | 7 / 5 px |
| Normality model | Threshold percentile | 99.5 by default (replaced by the validation-selected 3.553) |
| Normality model | Sampling | max 200,000 samples, seed 7 |

## Normality Model

We also implement a model to detect abnormal pixels in the images that may correspond to defects. To do this we implement a normality model based on the feature maps we have built.

We implement `fit_normal_model` which fits a per-channel Gaussian (mean and std) to feature vectors sampled from defect-free training images. It sets the anomaly threshold as a high percentile of the smoothed, border-cropped standardised distance scores on that same normal data, so the threshold fixes the expected false-positive rate on normal pixels. This threshold is only a default, and we later replace it with a value selected on validation data.

We implement `select_mask_threshold` which is used to select the model threshold based on the F1 score on validation data. This function takes in the score maps of the validation images, the correct masks and a number of candidates to test. Then the candidate thresholds are defined as the quantiles of all validation pixel scores at levels evenly spaced in $[0.5, 0.999]$, and a F1 score is calculated for each candidate threshold. The best candidate threshold is returned.

We also implement `predict_anomaly` which applies the fitted model to an image. This function calculates the score for each pixel in the image, smooths them with mean pooling, and returns a mask of the pixels that exceed the model threshold (with the border pixels always set to false). It also returns the pooled scores array as well as a score for the whole image, which is a percentile (defined in the configuration) of the pooled pixels (ignoring borders). 


## Results and Analysis Task A-C

### Task A results : material classifier

Accuracy is the fraction of images whose predicted material matches the true material:

$$
\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = y_i\right] = \frac{\sum_{c=1}^{C} TP_c}{N}
$$

where $N$ is the number of images, $C = 5$ is the number of material classes, $y_i$ and $\hat{y}_i$ are the true and predicted labels, and $TP_c$ is the number of correctly classified images of class $c$.

Macro F1 is the unweighted mean of the per-class F1 scores, so each material counts equally regardless of its number of images:

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
F1_c = \frac{2\,\text{Precision}_c\,\text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \qquad
\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} F1_c
$$

where $FP_c$ and $FN_c$ are the false positives and false negatives of class $c$.

We report the following accuracy and macro F1 scores on the test set:

| Subset | Images | Accuracy | Macro-F1 |
|:---:|:---:|:---:|:---:|
| **Overall** | **99** | **0.9495** | **0.9492** |
| **Normal** | **49** | **0.9184** | **0.9136** |
| **Defective** | **50** | **0.9800** | **0.9799** |


To further investigate these scores we can look at the confusion matrix where each cell $C_{i,j}$ is the number of examples of the $i$-th class predicted as $j$-th class (the diagonal are the correct predictions). Figure 11 shows this confusion matrix. We notice a total of 5 misclassifications : 2 grid predicted as carpet, 2 grid predicted as tile, 1 leather predicted as carpet. Grid is clearly the most misclassified class.

<p align="center"><img src="student_files/report/material_conf_mat.png" width="40%" alt="Material classifier confusion matrix on the test set."><br><em><strong>Figure 11.</strong> Material classifier confusion matrix on the test set.</em></p>


Figure 12 shows the confidence histogram (left) and the reliability diagram (right). The confidence histogram shows that the model is generally confident, with most predictions falling in the 0.8 to 1.0 bins. The reliability diagram groups predictions into confidence bins and compares the mean confidence of each bin with the actual accuracy in that bin (marker size is the number of predictions). A perfectly calibrated model would have all points on the diagonal of this plot. Most predictions are in the upper right (high confidence and accurate predictions), with a few bins under the diagonal (overconfident) and some over the diagonal (underconfident). 

<p align="center"><img src="student_files/report/confhist_material.png" width="75%" alt="Confidence histogram and reliability diagram of the material classifier on the test data."><br><em><strong>Figure 12.</strong> Confidence histogram and reliability diagram of the material classifier on the test data.</em></p>


Let's analyse two of the errors more closely to understand why they happened. 

**Error 1: Grid predicted as carpet**

In this first mistake the model classifies a grid example as a carpet example with very high confidence (94 %).

To understand which features influenced this prediction most heavily, we look at which features contribute most to the difference between the score for the true class and the score for the predicted class.

Recall what our feature vector looks like. Each pixel gets a vector of local channels (colour, Gabor energies, gradient energy, edge densities). Each channel is then summarised over the whole image by a few pooling statistics, and these summaries are concatenated into a single descriptor:

$$
\mathbf{x} = \big[\;
\underbrace{\text{mean}(\text{colour}), \dots, \text{mean}(\text{edge})}_{\text{mean of each channel}},\;
\underbrace{\text{std}(\text{colour}), \dots, \text{std}(\text{edge})}_{\text{std of each channel}},\;
\underbrace{\text{p90}(\text{colour}), \dots, \text{p90}(\text{edge})}_{\text{90th percentile of each channel}}
\;\big]
$$

So every entry of $\mathbf{x}$ is one summary statistic of one feature channel.

For a logistic regression classifier, the score (logit) of class $k$ for an image with standardized descriptor $\mathbf{z}$ is $s_k = \mathbf{w}_k^\top \mathbf{z} + b_k$, and the predicted class is the one with the highest score. The gap between the predicted class $p$ (carpet) and the true class $t$ (grid) is therefore

$$
s_p - s_t = \sum_{j} \underbrace{(w_{p,j} - w_{t,j})\, z_j}_{c_j} \; + \; (b_p - b_t),
$$

where $c_j$ is the contribution of descriptor entry $j$ to the error. Each channel is summarised by several pooled statistics (mean, std, p90), so we add the contributions of all statistics belonging to the same channel $d$:

$$
C_d = \sum_{s \in \{\text{mean},\,\text{std},\,\text{p90}\}} (w_{p,(s,d)} - w_{t,(s,d)})\, z_{(s,d)}.
$$

A positive $C_d$ means channel $d$ pushed the image towards the wrong class (carpet), and a negative $C_d$ means it pushed towards the true class (grid). The family-level bars are obtained by summing $C_d$ over all channels in the same feature family. [AI-DESIGN][AI-LANGUAGE][HUMAN-CHECK]


Figure 13 shows these values for this first misclassification. We see that the biggest causes of the misclassification are as follows: 

- Edge density per orientation: edges at different orientations, especially orientation bin 1 (gradient direction between 45° and 90°)
- Vertical Gabor features of different frequencies, in particular frequency 0.22
- Global edge density


<p align="center"><img src="student_files/report/err1_feat.png" width="85%" alt="Error 1 (grid predicted as carpet): feature contributions."><br><em><strong>Figure 13.</strong> Error 1 (grid predicted as carpet): feature contributions.</em></p>

We can now compare the most impactful feature maps of the misclassified example with the feature maps of the nearest well-classified examples of both the misattributed class (carpet) and the true class (grid), shown in Figure 14. Because the logistic regression classifier operates on channel-wise summary statistics (such as mean, std, and p90), its prediction depends directly on the overall activation magnitudes in feature space.

The most striking visual difference is in the vertical Gabor filter (`#2: gabor_f0.22_o0_p0.0`). It responds strongly across the correctly classified grid example, but shows very little activation for both the misclassified grid and the correctly classified carpet example. This loss of vertical response removes strong evidence for the "grid" class. At the same time, edge features—specifically `#1: edge_orientation_1` and `#3: edge_density`—show visibly stronger activations on the misclassified grid and carpet than on the correctly classified grid, yielding higher summary values that push the prediction toward "carpet".

We hypothesize that the main cause of this misclassification is the **rotation** of the grid. Because Gabor filters and orientation-specific edge features are not invariant to in-plane rotations, rotating the grid alters which channels trigger, depressing the vertical Gabor response while inflating non-target edge channels.

<p align="center"><img src="student_files/report/err1_featmap.png" width="35%" alt="Error 1 (grid predicted as carpet): feature map analysis."><br><em><strong>Figure 14.</strong> Error 1 (grid predicted as carpet): feature map analysis.</em></p>


**Error 2: Grid predicted as tile**

We compute the same figures for another mistake where the model predicted a grid example as a tile. This time the model is not as confident (only 52%); however, the predicted probability for the correct class (grid) is close to 0 so this is still an interesting case.

Figure 15 shows that in this case it is the edge features (global edge density and densities in orientations 2 and 3) that are mainly contributing to this misclassification.

<p align="center"><img src="student_files/report/err2_feat.png" width="85%" alt="Error 2 (grid predicted as tile): feature contributions."><br><em><strong>Figure 15.</strong> Error 2 (grid predicted as tile): feature contributions.</em></p>


In Figure 16 we can see that the results are not as clear cut as in the previous error and it is difficult to understand why this was a mistake, especially considering the similarity of the images with the correctly classified grid. However, we can still see that the edge channels have much stronger activations in the mistake example than on the correctly classified one, which brings it closer to a tile prediction.

<p align="center"><img src="student_files/report/err2_featmap.png" width="35%" alt="Error 2 (grid predicted as tile): feature map analysis."><br><em><strong>Figure 16.</strong> Error 2 (grid predicted as tile): feature map analysis.</em></p>



### Question 4: Does the model do worse on defective images?

Figure 17 shows accuracy on normal versus on defective images for each material class. The impact of defects seems to be minimal, and overall the model is actually more accurate on defective images (0.98) than on normal ones (0.92). This is mainly due to the grid class, which surprisingly gets 10/10 defective images right but only 6/10 good ones. These results are computed on a very small amount of data (10 images per bar), so they have to be taken with a grain of salt.

<p align="center"><img src="student_files/report/norm_vs_defect.png" width="50%" alt="Accuracy of the material classifier on normal vs defective images, per material."><br><em><strong>Figure 17.</strong> Accuracy of the material classifier on normal vs defective images, per material.</em></p>


Nevertheless, the model is slightly less accurate on defective leather images (9/10) due to a single error, which we investigate below.

Again, we plot the feature contributions and relevant feature maps (Figures 18 and 19) to understand why the model is wrong. However, the results are inconclusive, and it seems that factors independent of the defect (general colour, edges all over the image,...) are at play instead of a defect specific issue. 

<p align="center"><img src="student_files/report/leather_defect.png" width="85%" alt="Feature contributions for the misclassified defective leather image."><br><em><strong>Figure 18.</strong> Feature contributions for the misclassified defective leather image.</em></p>

Looking further at the specific feature maps causing the errors, it is again difficult to see the contribution of the defect to this misclassification.
<p align="center"><img src="student_files/report/leather_defect_featmap.png" width="45%" alt="Feature maps of the misclassified defective leather image."><br><em><strong>Figure 19.</strong> Feature maps of the misclassified defective leather image.</em></p>


### Task B results : Normality Model


After fitting one normal model per material on all of its (defect-free) training images, we tune the mask threshold on the validation data. To keep run time reasonable, Task B uses 2 images per material/defect group (including `good`), i.e. 60 validation and 60 test images, most of which contain defects. 

Figure 20 shows a histogram of the scores of good pixels and defect pixels as different distributions, as well as the threshold as a dashed line. Orange areas before the threshold indicate false negatives, whereas blue areas after the threshold indicate false positives. The log scale accentuates the long tail, but we do see that the separation between classes is very poor and a lot of validation error still remains.

<p align="center"><img src="student_files/report/val_thresh_hist.png" width="60%" alt="Histogram of the scores of good and defect pixels on the validation set, with the tuned threshold (dashed line)."><br><em><strong>Figure 20.</strong> Histogram of the scores of good and defect pixels on the validation set, with the tuned threshold (dashed line).</em></p>

We then freeze the best model based on validation data, and apply it to the test data to report the performance.

We report the following metrics: 
- **IoU (Intersection over Union)**: calculated as the number of correct mask pixels (the intersection of the predicted and true masks) divided by the number of pixels in either mask (their union). 

- **F1-Score**: as previously defined, this score is the harmonic mean of precision and recall; we calculate it pixel-wise. 

- **AUROC** (not requested but nice to include): The area under the receiver operating characteristic curve gives us an idea of how the model performs under all thresholds. It is the area under the curve where the x axis is the false positive rate, and the y axis is the true positive rate. A value of 0.5 is random guessing. This score is computed on image scores: it measures whether images as a whole can be classified as having a defect or not.

IoU and F1-Scores are calculated globally and for each type of defect, whereas AUROC is only calculated once, on the image scores (one per image), as it is a binary classification problem (defect or good).

The table below shows the test results, with defect types sorted by F1. The image AUROC is 0.69. Note that the F1 and IoU of the `good` images are 0 by construction, as they contain no defect pixels (they only contribute false positives to the overall score).

| Defect type | Pixel F1 | Pixel IoU |
|---|---|---|
| **Overall** | **0.175** | **0.096** |
| liquid | 0.650 | 0.481 |
| color | 0.381 | 0.236 |
| combined | 0.312 | 0.185 |
| glue | 0.278 | 0.162 |
| hole | 0.240 | 0.136 |
| fold | 0.121 | 0.065 |
| cut | 0.100 | 0.053 |
| poke | 0.032 | 0.016 |
| bent, broken, crack, glue_strip, gray_stroke, metal_contamination, oil, rough, scratch, thread | 0.000 | 0.000 |
| good | 0.000 | 0.000 |


First, on the global score, we notice that the AUROC is only moderately above random guessing at 0.69. This indicates the image scores do a poor job at separating good from defect images.

Figure 21 visualises the F1-score per defect class. 

<p align="center"><img src="student_files/report/f1_defect_class.png" width="65%" alt="Pixel-wise F1 score per defect class."><br><em><strong>Figure 21.</strong> Pixel-wise F1 score per defect class.</em></p>

We notice that the classes the model performs best on are those that constitute defects which form diffuse "blobs" such as liquid, colour and glue, whereas it struggles much more on defects such as scratch or crack which form thin, irregular connected edges (although the scratch results are noisy, see the full-data comparison below). We can hypothesise this is in part linked to the fact we pool pixel scores, meaning things like cracks and scratches which affect narrow areas get averaged out, or on the contrary the size of the defect is greatly overestimated.

Furthermore, the defects where the model performs the worst are mainly those attributed to the grid and tile textures, where the big variations naturally present in the texture confound the model.


Figure 22 shows examples of this:
- The top panel shows an easy example with a liquid defect that occupies a large area and is one diffuse component.
- The middle panel shows how a small fold can be greatly overestimated due to pooling.
- The lower panel shows a worst case scenario where the defect is small at any given point (crack) and the material is very irregular (tile), leading the model to fail to detect any defect at all.

<p align="center"><img src="student_files/report/defect_seg_examples.png" width="50%" alt="Some results of the normality model on test examples."><br><em><strong>Figure 22.</strong> Some results of the normality model on test examples.</em></p>

### Question 5: Border failure and threshold impact

The border exclusion we use (the outer pixels are ignored) is important to prevent artefacts from cross correlation and pooling, which are unreliable near the image edges. However, it also means we are unable to detect defects near the borders, and our masks cut off there too. The example in Figure 23 illustrates this: notice how the mask gets sharply cut off due to the border exclusion, even though those pixel scores were high.

<p align="center"><img src="student_files/report/border_failure.png" width="85%" alt="Example of the border exclusion causing an early cutoff of a defect."><br><em><strong>Figure 23.</strong> Example of the border exclusion causing an early cutoff of a defect.</em></p>

Assuming the aggregation percentile is the percentile used to define the whole image score, increasing it would mean that image scores increase or stay the same, as the score moves closer to the maximum pixel score. This makes the image score more sensitive to small defects, but also to isolated noisy pixels in good images. The mask is computed separately, so it would not change. The AUROC is computed over all thresholds, but it could still change, since images do not all increase by the same amount and their ranking can change.

Increasing the mask threshold means that pixels have to have higher scores to be considered defects. This generally increases precision as it is more difficult for noise to be considered defects, but it also reduces recall as we miss smaller or faint defects that were detected before. The score map and image score do not change.

### Ablations: Gabor only and edge only 
We also compare the Gabor-only, edge-only and combined feature maps, keeping all other settings identical. Each branch goes through the same validation-to-test sequence (threshold selected on validation, then frozen and applied to test).

| Feature branch | Selected threshold | Test pixel F1 |
|---|---|---|
| Gabor only | 2.695 | 0.146 |
| Edge only | 3.592 | 0.164 |
| **Combined** | **3.553** | **0.175** |

Overall, the Gabor-only branch performs the worst and the edge-only branch already does better, while combining both gives the best F1. The thresholds cannot be compared directly between branches, as each score is computed over a different set of channels. Note also that "Gabor only" excludes colour, while "Combined" includes the RGB channels.

The overall F1 hides clear differences between defect types. Although the branches use different feature channels, each one produces a single anomaly score per pixel, so their maps can be compared on the same images. In Figure 24, each map is divided by its branch's threshold so they share one colour scale.

<p align="center"><img src="student_files/report/ablation_maps.png" width="75%" alt="Gabor-only, edge-only and combined anomaly maps on three wood test images, normalised by each branch's threshold, with the predicted masks outlined."><br><em><strong>Figure 24.</strong> Gabor-only, edge-only and combined anomaly maps on the same test images. Each map is divided by its branch's threshold (1 = threshold), and the predicted mask is outlined in cyan.</em></p>

- **Edge only** is the best branch on the largest number of defect types (colour, cut, fold, glue, poke), and by far on colour defects (F1 0.57, against 0.22 for Gabor only and 0.38 combined). The stains have sharp borders, so they create edges. In the wood colour example, the edge-only masks have the right size, while the Gabor and combined masks are greatly oversized.
- **Gabor only** is the only branch that detects scratches (0.35 against 0 for both others) and the thin tile marks (glue_strip, gray_stroke). Combining dilutes this signal: the combined score averages over all 13 channels, so a response in a few Gabor channels falls below the threshold (wood scratch row).
- **Combined** wins where both branches contribute, such as liquid, the "combined" defect type, and the wood hole example (F1 0.49 against 0.37 and 0).
- No branch detects the grid defects, the tile cracks, or the carpet cuts and holes.

The per-defect values rely on only 2 test images each, so they are indicative. They still show that the branches are complementary, and that averaging all channels into one score can hide a defect that a single branch sees.

### Effect of using the full data (Tasks A and B)

All the results above use the small deterministic subsets of the notebook:

- **Task A:** 4 training images per material, evaluated on 10 good and 10 defective test images per material.
- **Task B:** 2 images per material/defect group for validation and test.

To see how much our conclusions depend on this, we repeat the same pipeline on the complete splits: all 1266 training images for Task A, and all 267 validation and 248 test images (65 of them good) for both tasks. Everything else is unchanged: features, pooled statistics, $C = 5$, and the normal models, which were already fitted on all training images. No setting is tuned on these test results. The results are summarised in the tables below and in Figure 25. The frozen models used for the personal photos remain the ones trained on the subset.

| Task A run | Accuracy | Macro-F1 | Accuracy good | Accuracy defective |
|---|---|---|---|---|
| 20 train, 99 test (reported above) | 0.949 | 0.949 | 0.918 | 0.980 |
| 20 train, full test | 0.940 | 0.929 | 0.938 | 0.940 |
| Full train, full test | **1.000** | **1.000** | **1.000** | **1.000** |

| Task B run | Selected threshold | Pixel F1 | Pixel IoU | Image AUROC |
|---|---|---|---|---|
| 2 per group (reported above) | 3.553 | 0.175 | 0.096 | 0.690 |
| Full validation and test | 3.402 | 0.168 | 0.092 | 0.662 |

<p align="center"><img src="student_files/report/full_data_comparison.png" width="95%" alt="Full-data Task A confusion matrix and Task B pixel F1 per defect type for the subset and the full test split."><br><em><strong>Figure 25.</strong> Left: Task A confusion matrix with the classifier trained on all training images and evaluated on the full test split. Right: Task B test pixel F1 per defect type, on the 2-per-group subset and on the full test split.</em></p>

**Task A.** The model trained on 20 images keeps roughly the same accuracy on the full test split (0.940 instead of 0.949), so our subset results were representative of that model. Training on all 1266 images, however, makes every one of the 248 test images correct. All the errors analysed above therefore come from the tiny training set rather than from a limit of the descriptor. With 4 images per material, a rotated grid or a darker leather sample is simply outside what the classifier has seen, while the full training set covers enough of this variation. The difference between good and defective images (Question 4) also disappears, which confirms that it was noise rather than an effect of the defects. Note that 5 well-separated MVTec materials captured under one controlled setup make this an easy problem. The personal photos show that this does not carry over to different capture conditions, and they were classified with the 20-image model.

**Task B.** Here only the evaluation sets change, and the overall results barely move: pixel F1 goes from 0.175 to 0.168, and image AUROC from 0.69 to 0.66. The global conclusions hold. Diffuse defects (liquid, color, combined, glue) remain the easiest, and most grid and tile defects (bent, broken, crack, glue_strip, gray_stroke, oil, metal_contamination) stay at or near 0. The per-defect values, however, are noisy on the subset, as they rely on only 2 images each. The main example is scratch: it scored 0 on the subset but reaches 0.58 on the full test split, so our earlier claim that scratches are hard to detect only holds for the two scratches we happened to evaluate. Thread also goes from 0 to 0.09, while hole drops from 0.24 to 0.13. The per-defect comparison should therefore be read as indicative only.

### Task C: DTD Attribute Prediction

In this part we build a model that predicts appearance attributes for images from the DTD dataset. 

The first step is preparing the dataset (520 training and 520 validation images, 40 per primary attribute, resized to 64×64) and extracting handcrafted features from the previous part using the `extract_local_features` and `global_pool` functions.

We then apply 2 component PCA to visualise our data in two dimensions. We fit the PCA on the training and validation data but only visualise the training examples on the plot. The original pipeline does not standardise the data, but seeing the cone pattern that emerged, we decided to also standardise the data to compare. However, the non-standardised descriptors are used for the rest of the section to respect the instructions (the classifier pipeline standardises them internally).

Figure 26 shows both the un-standardised and standardised PCA projections of the training data. As we can see, the classes are not well separated in the projection, suggesting they may be difficult to separate linearly.

<p align="center"><img src="student_files/report/dtd_pca.png" width="80%" alt="PCA projection of the DTD training and validation examples, with non-standardised and standardised features."><br><em><strong>Figure 26.</strong> PCA projection of the DTD training and validation examples, with non-standardised and standardised features.</em></p>

We then fit a one-vs-rest classifier on the training data and evaluate it on the validation set. We report two metrics on the validation set:
- **Average Precision (AP)**: the area under the precision-recall curve for one attribute. It measures how well the model ranks positive images above negative ones, independently of any threshold. A random model gets an AP equal to the proportion of positives (here around 0.08 to 0.13).
- **Macro F1**: the F1 score is computed for each attribute separately with a fixed threshold of 0.5, then averaged so every attribute counts equally, no matter how many positives it has.

| Attribute | AP | F1 (threshold 0.5) | Positives |
|---|---|---|---|
| striped | **0.541** | **0.519** | 51 |
| banded | 0.442 | 0.500 | 41 |
| woven | 0.332 | 0.179 | 46 |
| grid | 0.256 | 0.237 | 41 |
| marbled | 0.211 | 0.045 | 43 |
| blotchy | 0.197 | 0.000 | 70 |
| stained | 0.178 | 0.067 | 41 |
| bumpy | 0.159 | 0.043 | 43 |
| braided | 0.143 | 0.000 | 42 |
| fibrous | 0.131 | 0.000 | 43 |
| cracked | 0.122 | 0.000 | 40 |
| porous | 0.111 | 0.000 | 45 |
| pitted | 0.081 | 0.000 | 43 |
| **Macro average** | **0.223** | **0.122** | |

Figure 27 summarises performance per attribute: 

<p align="center"><img src="student_files/report/dtd_per_attr.png" width="85%" alt="Per-attribute performance of the attribute model on the DTD validation set."><br><em><strong>Figure 27.</strong> Per-attribute performance of the attribute model on the DTD validation set.</em></p>

We notice that the attributes the model handles best are striped, banded, woven and grid, which are all defined by a regular orientation or repetition. This is exactly what our Gabor filters (two fixed frequencies at 0° and 90°) and orientation-specific edge densities measure, so the mapping from features to attribute works well here. On the other hand, attributes such as pitted, porous, cracked or fibrous are close to chance level. These are described by irregular small-scale structure or 3D relief, which our global descriptor (pooled statistics over a 64×64 image) does not capture well: the spatial arrangement is lost by pooling, and the fine detail is lost by resizing. Blotchy, stained and marbled sit in between, probably because they rely partly on colour variations, which our RGB channels do capture to some degree.

The F1 scores are much lower than the AP, and many are exactly 0. This is because each attribute is positive for only around 10% of the images, so the predicted probabilities rarely go above the 0.5 threshold, even when the ranking (AP) is somewhat better than chance. A threshold tuned per attribute would likely improve F1 considerably.

Figure 28 shows the probability heatmap, where each row is a validation example and each column is an attribute. The cells correspond to the predicted probabilities for a given example and a given class. The red squares are the true attributes of the examples.

<p align="center"><img src="student_files/report/proba_heatmap.png" width="55%" alt="Predicted probability heatmap for the DTD validation examples (rows) and attributes (columns)."><br><em><strong>Figure 28.</strong> Predicted probability heatmap for the DTD validation examples (rows) and attributes (columns).</em></p>

We notice that the probabilities are low overall, with only four cells reaching 0.5, which explains the low F1 scores at the 0.5 threshold. The model only gives its highest probability to the correct attribute for a few examples (banded, blotchy, grid and woven), which are mostly the regular, oriented textures we identified before. When it is wrong, the confusions are often understandable: the braided example is predicted as cracked, the bumpy example (a pattern of blocks with strong straight edges) as banded, and the marbled example as woven. The striped example is a good illustration of the limits of our features: its stripes are curved and diagonal, so they do not match our Gabor filters at 0° and 90°, and the model predicts stained instead with 0.50. Finally, some examples have several true attributes (banded is also striped, woven is also grid), and the model generally only picks up one of them.


### Question 6: What information is present or absent in handcrafted descriptors?

The handcrafted descriptors indicate which types of patterns are present in the image and how strongly. The Gabor energies measure how much repetitive stripe structure there is at each of our fixed frequencies and orientations. The gradient energy and edge densities summarise how many edges there are and which orientations dominate. The RGB channels capture the overall colour and how much it varies. Because we pool the mean, standard deviation and percentiles, the descriptor keeps both the average level of each channel and its extreme activations.

However, because we use summary statistics, we lose most of the information about where these patterns appear and how they are arranged relative to each other. For example, a regular grid and the same edges scattered randomly can give very similar descriptors. Each channel is also pooled independently, so the descriptor does not know whether two features occur at the same location. Finally, some information is never measured at all: orientations and frequencies outside our Gabor filter bank (such as diagonal or curved stripes) are only seen indirectly through the coarse edge-orientation bins. This explains why regular, oriented attributes (striped, banded, grid) are predicted well while irregular or relief-based ones (pitted, porous, cracked) are close to chance.


## Results and Analysis Task D

In this section we use a Vision Language Model (VLM) to predict defects. The chosen model is GPT-5.6 Sol.

For generic zero-shot (D1) we use the following prompt: 

```
Inspect this surface image. Is anything visibly abnormal? 
Return only valid JSON withis_defective, defect_name, confidence, and evidence.
```

For named-defect zero-shot (D2) we use the following prompt (example for one of the queries): 

```
Inspect this surface image and find if there is among the following :
- color: A localized region has an abnormal colour or tone.
- cut: A sharp incision or sliced region interrupts the texture.
- hole: Material is missing in a compact hole-like region.
- metal_contamination: A metallic foreign object lies on the surface.
- thread: A foreign thread lies across the regular texture.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```


For example-conditioned (D3), we use the same prompt as in named-defect zero-shot, but we append the following to the end, and attach the supplied validation support image:

```
One of the attached image provides examples for each type of defect, use it to establish the defect type.
```

The results are as follows:

**D1: Generic zero-shot**

| Query | Material | Defective | Predicted label | Confidence |
|:---:|:---:|:---:|:---|:---:|
| Q1 | carpet | Yes | localized weave distortion/damaged fibers | 0.98 |
| Q2 | grid | Yes | broken mesh strands | 0.99 |
| Q3 | leather | Yes | puncture/tear | 0.99 |
| Q4 | tile | Yes | crack | 0.99 |
| Q5 | wood | Yes | split/crack | 0.98 |

**D2: Named-defect zero-shot**

| Query | Material | Defective | Predicted label | In vocabulary | Confidence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q1 | carpet | Yes | color | Yes | 0.96 |
| Q2 | grid | Yes | broken | Yes | 0.99 |
| Q3 | leather | Yes | poke | Yes | 0.97 |
| Q4 | tile | Yes | crack | Yes | 0.99 |
| Q5 | wood | Yes | scratch | Yes | 0.98 |

**D3: Example-conditioned**

| Query | Material | Defective | Predicted label | In vocabulary | Confidence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q1 | carpet | Yes | color | Yes | 0.96 |
| Q2 | grid | Yes | broken | Yes | 0.98 |
| Q3 | leather | Yes | cut | Yes | 0.96 |
| Q4 | tile | Yes | crack | Yes | 0.99 |
| Q5 | wood | No | good | Yes | 0.97 |


- Compare D1 anomaly discovery with D2/D3 exact-label accuracy.

The D1 labels are plausible descriptions of the defects, while mostly not matching the vocabulary exactly. Note that for query 4 it did match an exact vocabulary word (crack), and the label for query 2 ("broken mesh strands") contains the correct vocabulary word (broken).

- Count answers that violate the allowed vocabulary.

In D1 only one out of five answers is in the vocabulary, but this is expected. In D2 and D3 none of the predicted labels fall outside of the vocabulary, meaning for these queries the VLM followed that instruction correctly.

- Plot confidence for correct and incorrect predictions by condition.

We use the labels in the solution (`solution/task_d_answer_key.csv`) to compare to the output of the VLM after the answers were frozen.  For D1, since it doesn't predict exact labels by design, we only report defect presence prediction. Figure 29 shows the result. 

<p align="center"><img src="student_files/report/vlm_conf.png" width="45%" alt="VLM confidence for correct and incorrect predictions, by condition."><br><em><strong>Figure 29.</strong> VLM confidence for correct and incorrect predictions, by condition. [AI-VISION] [HUMAN-CHECK]</em></p>

In terms of accuracy per condition, D1 correctly flags all 5 queries as defective (5/5 defect presence, exact labels are not scored), D2 gets 4/5 exact labels (Q3 leather is predicted as poke instead of cut), and D3 also gets 4/5 exact labels (Q5 wood is predicted as good instead of scratch). 


Overall, all of the confidences are quite high, ranging from 0.96 to 0.99. There seems to be little correlation between confidence and accuracy: the two incorrect predictions (D2 Q3 and D3 Q5) both have a confidence of 0.97, which is right in the middle of the correct ones. Confidence is slightly lower when more information is given (mean of 0.986 for D1, 0.978 for D2 and 0.972 for D3), but these differences are very small and only computed on 5 queries each. One explanation may be that the model cannot exactly match the defect with the definitions or examples, so it reports a lower confidence.

- Identify queries where definitions help but examples do not, and vice versa.

For query 5 (scratched wood), adding the definitions allowed the model to correctly predict that the defect is a scratch, when it had predicted it as a split/crack without them. But interestingly, when we add an example, it fails to detect any defect at all and describes the scratch as a "natural seam-like boundary". This is likely because the scratch example provided in the support set has a specific shape and does not look like the one we are testing on.

For the cut leather example however, adding an example made the model go from incorrectly predicting a poke to correctly predicting a cut.

- Discuss prompt sensitivity, model updates, nondeterminism, and possible prior exposure to the public MVTec dataset.

The prompt greatly influences how a model responds. For example, had the prompts put more emphasis on accurately reporting confidence, the confidences may have been better correlated with the results.

Model updates are also a problem for reproducibility. Models behind a commercial interface such as ChatGPT are regularly updated or replaced by the provider, sometimes without any visible change in name. This means the same prompt and image may give a different answer a few weeks later, and our results only hold for the model we used (GPT-5.6-sol) on the date we ran the queries (2026-10-04). This is why we record the model version and run date with every prediction.

VLMs are nondeterministic, as they sample their output from a probability distribution over the next tokens and generate autoregressively. Theoretically, setting the temperature to 0 makes the model deterministic, meaning that given the same input it returns the same output. However, in practice floating-point operations are not associative, and the order in which they are computed on the GPU can change between runs (for example depending on how requests are batched together), so small numerical differences can still change the chosen token. This may be a problem for some applications, as the same image can produce different outputs.

Finally, VLMs are pretrained on massive amounts of data, much of it scraped from the internet, and then fine-tuned on smaller curated datasets. MVTec AD is a popular public dataset, so it is not unlikely that it was included in one of those stages. This would be a form of data leakage: the model may recognise the test images or their labels rather than actually reasoning about the defect, which would make the results look better than they would be on new images.

This small experiment showcases the capabilities of VLMs as versatile models, but also their limits, as the quality of their answers depends on the prompt and may vary in unexpected ways. However, this experiment is too small to draw any general conclusions.


## Results and Analysis Custom Photos

In this section we analyse the results of our models on 17 custom photos spanning three MVTec classes: wood, tile and grid. Each photo is cropped automatically via a shell script.

**Data collection.** The 17 photos cover grid (4), tile (10) and wood (3), and their EXIF metadata was removed. As I am working alone, I am the sole annotator: the 3 to 4 DTD terms of each photo were recorded in the manifest before any prediction was viewed. The capture conditions vary in illumination (wood_bad_light is side-lit compared to wood, and tile_black and tile_white_hole are under direct light), in scale (tile_grid is taken from far away, and grid_round_far is a more distant shot of the same grid as grid_round_close) and in viewpoint (grid_round_far is also slightly tilted; tile_granite_tilted is an oblique shot of the same tile as tile_granite). Four photos contain anomalies: a colour change (tile_colour), a liquid (tile_liquid) and two holes (tile_white_defect, tile_white_hole). [HUMAN-DATA]

For each photo we extract feature maps as we have done previously and predict the material class using the frozen material classifier, and the top-5 DTD attributes using the frozen Task C attribute model. The normality model for the predicted material class is then applied to score the image and segment defects. The information is then gathered into cards known as texture "passports".

**Note :** For the descriptions and uncertainty fields, a VLM (Claude Opus 5.5) was used to initially annotate the photos, thus this is a [AI-VISION][AI-LANGUAGE][HUMAN-CHECK] piece of the assignment, with the data used being [HUMAN-DATA].

We group the passports by how similar they look to the MVTec training images. This grouping is a subjective visual judgement on our part rather than a measured distance (the measured distances are analysed at the end of this section):

**Close to training examples (Figure 30):**

<p align="center"><img src="student_files/report/passports/wood_pale.png" width="90%" alt="Texture Passport of wood pale"><br><small><strong>(a)</strong> wood pale</small><br>
<img src="student_files/report/passports/wood.png" width="90%" alt="Texture Passport of wood"><br><small><strong>(b)</strong> wood</small><br>
<img src="student_files/report/passports/grid_round_close.png" width="90%" alt="Texture Passport of grid round close"><br><small><strong>(c)</strong> grid round close</small><br>
<img src="student_files/report/passports/tile_white_defect.png" width="90%" alt="Texture Passport of tile white defect"><br><small><strong>(d)</strong> tile white defect</small><br>
<img src="student_files/report/passports/tile_white_hole.png" width="90%" alt="Texture Passport of tile white hole"><br><small><strong>(e)</strong> tile white hole</small><br>
<img src="student_files/report/passports/tile_granite.png" width="90%" alt="Texture Passport of tile granite"><br><small><strong>(f)</strong> tile granite</small><br>
<em><strong>Figure 30.</strong> Texture Passports of the photos judged close to the training examples: (a) wood pale, (b) wood, (c) grid round close, (d) tile white defect, (e) tile white hole, (f) tile granite.</em></p>

**Further from training examples (Figure 31):**

<p align="center"><img src="student_files/report/passports/grid_round_far.png" width="90%" alt="Texture Passport of grid round far"><br><small><strong>(a)</strong> grid round far</small><br>
<img src="student_files/report/passports/grid_square.png" width="90%" alt="Texture Passport of grid square"><br><small><strong>(b)</strong> grid square</small><br>
<img src="student_files/report/passports/tile_black.png" width="90%" alt="Texture Passport of tile black"><br><small><strong>(c)</strong> tile black</small><br>
<img src="student_files/report/passports/wood_bad_light.png" width="90%" alt="Texture Passport of wood bad light"><br><small><strong>(d)</strong> wood bad light</small><br>
<img src="student_files/report/passports/tile_granite_tilted.png" width="90%" alt="Texture Passport of tile granite tilted"><br><small><strong>(e)</strong> tile granite tilted</small><br>
<em><strong>Figure 31.</strong> Texture Passports of the photos judged further from the training examples: (a) grid round far, (b) grid square, (c) tile black, (d) wood bad light, (e) tile granite tilted.</em></p>

**Very far from training examples (Figure 32):**

<p align="center"><img src="student_files/report/passports/tile_noliquid.png" width="90%" alt="Texture Passport of tile noliquid"><br><small><strong>(a)</strong> tile noliquid</small><br>
<img src="student_files/report/passports/tile_no_colour.png" width="90%" alt="Texture Passport of tile no colour"><br><small><strong>(b)</strong> tile no colour</small><br>
<img src="student_files/report/passports/tile_colour.png" width="90%" alt="Texture Passport of tile colour"><br><small><strong>(c)</strong> tile colour</small><br>
<img src="student_files/report/passports/tile_liquid.png" width="90%" alt="Texture Passport of tile liquid"><br><small><strong>(d)</strong> tile liquid</small><br>
<img src="student_files/report/passports/grid_chair.png" width="90%" alt="Texture Passport of grid chair"><br><small><strong>(e)</strong> grid chair</small><br>
<img src="student_files/report/passports/tile_grid.png" width="90%" alt="Texture Passport of tile grid"><br><small><strong>(f)</strong> tile grid</small><br>
<em><strong>Figure 32.</strong> Texture Passports of the photos judged very far from the training examples: (a) tile noliquid, (b) tile no colour, (c) tile colour, (d) tile liquid, (e) grid chair, (f) tile grid.</em></p>

Overall, the models seem to struggle the most with examples further from the training data, i.e. examples of materials that don't closely match the MVTec style. As soon as the model predicts the wrong class, anomaly detection becomes very difficult, as it uses the wrong normality model. Even on materials very similar to the training data (e.g. the wood_pale example), and using a normality model fitted to the correct material, lighting still greatly impacts the image score and leads to false positive defect masks.

We look at the agreement between human annotations of DTD attributes and model predictions. As I am the sole annotator, a human–human comparison is not possible, so we only compare human and model terms. For a given image, we take its $K$ human annotations and the top-$K$ attributes scored by the model. Figure 33 shows the overlaps. We notice that the biggest agreement is on the marbled attribute (8 photos), meaning the model's notion of this term may closely line up with the human one. Blotchy comes second with 6 agreements. 

<p align="center"><img src="student_files/report/personal_attribute_overlap.png" width="70%" alt="Overlap between the human DTD terms and the model's top-K attributes for each personal photo."><br><em><strong>Figure 33.</strong> Overlap between the human DTD terms and the model's top-K attributes for each personal photo.</em></p>


Figure 34 plots the material confidence against the anomaly image score. Confidence says nothing about whether the anomaly score can be trusted: every image score above 5 comes from a photo whose material was predicted wrongly, sometimes with a confidence above 0.9 (tile_liquid, tile_grid, tile_no_colour). These high scores therefore mostly measure how different the photo is from the wrong normal model, not real defects. Conversely, the two photos with real holes (tile_white_defect, tile_white_hole) are not separated from the defect-free ones: tile_white_defect, whose material is correct, stays below the threshold.

<p align="center"><img src="student_files/report/personal_conf_vs_score.png" width="50%" alt="Material classifier confidence vs anomaly image score for the personal photos."><br><em><strong>Figure 34.</strong> Material classifier confidence vs anomaly image score for the personal photos.</em></p>

Given we are likely working with out of distribution images, it is interesting to see where they lie in a PCA with the training data. This PCA only explains 74% of the variance of the training data, so distances may not be accurate but it does give some intuition over why our images were misclassified. Figure 35 shows this. The red stars represent misclassified personal pictures, whereas the green stars are correctly classified.

<p align="center"><img src="student_files/report/personal_ood_pca.png" width="50%" alt="PCA of the material classifier training descriptors, with the MVTec test images and the personal photos projected onto it."><br><em><strong>Figure 35.</strong> PCA of the material classifier training descriptors, with the MVTec test images and the personal photos projected onto it.</em></p>

We notice that the misclassified images are generally far from the material clusters, and those that are correctly classified are closer to the training data of the correct cluster. Interestingly, the grid_square which was predicted as tile with high confidence does seem to be inside the tile cluster. The two granite photos sit close to each other and between the carpet and tile clusters: tile_granite_tilted is correctly classified as tile but with a confidence of only 0.34, and tile_granite is predicted as carpet with 0.46. Tilting the camera barely moves the photo in this projection, but in the full standardised space the nearest-training-image distance grows from 5.4 (tile_granite, the closest of all personal photos) to 6.7, so the correct label of the tilted photo looks fragile rather than a sign of being close to the tile training data.

Finally, we also analyse if confidence drops with distance to training data. Since logistic regression behaves like a smooth step function, it can produce very high-confidence regions even where there is no training data. Figure 36 shows that this seems to be the case: the model confidently predicts examples which are very far from the training data. On the MVTec test images, confidence only drops slightly with distance (Spearman correlation of −0.21; mean confidence of 0.97 below a distance of 2 and 0.90 beyond the 95th percentile at 7.1), while the error rate rises from 0% to 62%. On the personal photos there is no significant correlation (Spearman 0.28, p = 0.28, n = 17), and if anything it is positive rather than negative. Confidence is therefore a poor warning sign, whereas the distance to the training data itself would be a useful out-of-distribution flag.

<p align="center"><img src="student_files/report/personal_ood_distance.png" width="50%" alt="Material classifier confidence vs distance to the nearest training image, for the MVTec test images and the personal photos."><br><em><strong>Figure 36.</strong> Material classifier confidence vs distance to the nearest training image, for the MVTec test images and the personal photos.</em></p>

**Would more training data help?** Our frozen material classifier was trained on only 20 images, so one might expect a better model to fix the personal photos. We refit it on all 1266 training images (same features and settings) and compare it with the frozen model. The passports still use the frozen model.

| | 20-image model (frozen) | Full-data model |
|---|---|---|
| MVTec test accuracy | 0.940 | 1.000 |
| Personal photos correct | 5 / 17 | 4 / 17 |
| Mean confidence on personal photos | 0.78 | 0.90 |
| Mean confidence when wrong | 0.78 | 0.89 |
| Photos beyond the MVTec-test 95th percentile distance | 10 / 17 | 17 / 17 |

The full-data model is perfect on MVTec but does not do any better on our photos. Only the two granite photos change label, and both become grid, which costs the tilted one its correct label. The wrong predictions also become much more confident (e.g. wood_bad_light goes from tile at 0.68 to tile at 0.98). With more training data, the MVTec images form tighter clusters, so all our photos now fall outside the region they cover. The failures are therefore caused by the change of capture conditions (a domain shift), not by a lack of training data, and more MVTec data only makes the model more confidently wrong.

**Reflection: what changes when a single factor changes.** Several of our photos come in pairs where only one factor differs: illumination, scale or viewpoint, or the presence of an anomaly. Comparing the two photos of each pair (table below, values from the passports) isolates the effect of that factor.

| Pair | Factor changed | Predicted material (confidence) | Image score |
|---|---|---|---|
| wood → wood_bad_light | illumination (side light) | wood (1.00) → tile (0.68) | 3.57 → 3.19 |
| tile_white_defect → tile_white_hole | illumination (direct light) | tile (0.55) → wood (0.78) | 3.06 → 40.87 |
| grid_round_close → grid_round_far | scale and viewpoint | grid (0.97) → wood (0.78) | 1.52 → 16.27 |
| tile_granite → tile_granite_tilted | viewpoint | carpet (0.46) → tile (0.34) | 4.46 → 1.72 |
| tile_no_colour → tile_colour | colour anomaly added | wood (0.97) → wood (0.98) | 10.47 → 8.45 |
| tile_noliquid → tile_liquid | liquid anomaly added | carpet (0.50) → wood (0.91) | 3.58 → 15.90 |

Changing a single acquisition factor is enough to change the predicted material in all four illumination, scale and viewpoint pairs. The side light on the wood creates a strong artificial edge, the direct light on the white tile creates a reflection, and moving away from the round grid changes the apparent frequency of its holes. Our descriptor depends directly on all of these, since it measures intensity, edge orientation and Gabor energy at fixed frequencies. The material classifier never saw such variations, because the 20 MVTec training images are all taken under the same controlled setup. The tilted granite photo is only classified correctly by chance, with a confidence of 0.34.

The anomaly pairs show the second weakness. Adding the colour anomaly does not raise the image score: it actually goes down (10.47 → 8.45), because the clean tile is already very far from the wood normal model it is wrongly compared to. Adding the liquid does raise the score, but the predicted material also changes (carpet → wood), so the comparison is made against a different normal model and the increase cannot be attributed to the liquid. In both cases the anomaly score measures the distance to the wrong normal model rather than the defect. A real defect is only flagged reliably when the material is right and the photo is close to MVTec conditions, which none of our anomaly photos satisfies.

If we were to redo this study, we would fix the capture setup (distance, angle and diffuse light) to match MVTec as closely as possible, and photograph each anomaly next to a clean reference of the same surface under identical conditions. We would then vary one factor at a time with more steps (for example three distances and three angles), to see how far the models can be pushed before they fail. On the model side, three changes would help:

- adding training images taken under more varied conditions (more MVTec images alone does not help, as shown above);
- making the descriptor less sensitive to lighting (for example normalising each image's intensity);
- refusing to produce a passport when the photo is too far from the training data, using the nearest-training-image distance shown above.


## Report traceability table

| Student file | Required sections | Evidence location |
|---|---|---|
| `gabor_branch.py` | `make_gabor_bank`, `gabor_energy_maps` | Report: Gabor Implementation (Implementation, Question 1; Figures 1–5). Notebook 1: section 3 (bank construction and kernel display, cells 8–9), section 4 (energy maps and orientation selectivity, cells 11–18). Unit tests: `test_gabor.py`. |
| `edge_branch.py` | `bilinear_sample`, `nms_interpolated`, `adaptive_thresholds`, `hysteresis`, `detect_edges` | Report: Edge Implementation (bilinear interpolation formula, robust/percentile threshold formula, Question 2; Figures 6–8). Notebook 1: section 5 (edge stages, cell 20; interpolated vs nearest NMS, cells 21–24; 4- vs 8-connectivity, cell 27). Unit tests: `test_edges.py`. |
| `features.py` | `extract_local_features`, `global_pool`, `feature_family_indices` | Report: Features and Representations (channel families, pooled descriptor, Question 3, Configuration table; Figures 9–10), Task A error analysis (per-family contributions, Figures 13–16, 18), Task C (Figure 26). Notebook 1: section 6 (descriptor audit and channel scales, cells 29–33). Notebook 2: section 3 (Task A descriptors; test accuracy 0.949, macro-F1 0.949). Notebook 3: section 3 (DTD descriptors and PCA). Unit tests: `test_features_normality.py`. |
| `normality.py` | `fit_normal_model`, `predict_anomaly`, `select_mask_threshold` | Report: Normality Model; Task B results (Figures 20–22, per-defect F1/IoU table; frozen threshold 3.553, pixel F1 0.175, IoU 0.096, image AUROC 0.69), Question 5 (Figure 23), Ablations (Figure 24). Notebook 2: sections 4–7 (normal model fit, validation threshold selection, frozen test results, feature-family ablation). Unit tests: `test_features_normality.py`. |

## AI-use disclosure table

| Tool / model | Affected artefacts | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| Claude Opus 5.5 (Claude Code) | `texturelab/edge_branch.py`: `bilinear_sample`, `hysteresis`; `texturelab/features.py`: `extract_local_features` | [AI-CODE] [HUMAN-CHECK] | I wrote these functions by hand first, then used the tool to debug them (index clipping in the bilinear sampler, the traversal in hysteresis, channel alignment and naming in the feature extractor). | Full unit-test suite passes (`python -m unittest discover -s tests`, 23 tests); outputs inspected visually in Notebook 1 (NMS, hysteresis and 4/8-connectivity figures) and channel names and shapes checked in Notebooks 1–3. |
| Claude Opus 5.5 (Claude Code) | Notebook 1 cells 17 (pooling-size example for Q1) and 27 (4- vs 8-connectivity comparison for Q2) | [AI-CODE] | Plotting and helper code for the figures used to answer Q1 and Q2. | Cells run from a fresh kernel; figures checked against the input images and against the behaviour expected from the implemented functions. |
| Claude Opus 5.5 (Claude Code) | Notebook 2 cells 11 (confidence histogram, reliability diagram), 13 and 16 (error analysis and response maps), 15 (good vs defective accuracy), 27 and 32 (defect example helpers and figure), 35 (storing the ablation models), 37 (branch-map comparison, Figure “ablation_maps”), 40 (full-data run, Figure “full_data_comparison”) | [AI-CODE] | Evaluation, plotting and figure code around the required pipeline (routine plotting grouped in one row). The model fitting, threshold selection and metrics use the course pipeline and my `texturelab` implementations. | Cells run from a fresh kernel; the numbers quoted in the report were checked against the printed cell outputs; the ablation keeps the same pooling, training images and validation-to-test sequence for every branch. |
| Claude Opus 5.5 (Claude Code) | Report, Task A error analysis (logit-contribution decomposition of the confident grid→carpet error, equations for $c_j$ and $C_d$) | [AI-DESIGN] [AI-LANGUAGE] [HUMAN-CHECK] | Proposed decomposing the logit difference into per-channel contributions and helped phrase the explanation. | Contributions recomputed in Notebook 2 (cells 13/16) and inspected by me |
| Claude Opus 5.5 (Claude Code) | `student_files/crop_photos.sh`, `student_files/annotate.ipynb` | [AI-CODE] | Script to centre-crop, resize and strip EXIF from the personal photos; small notebook to record the DTD terms of each photo into the manifest. | Cropped photos inspected; EXIF absence checked on the output files; manifest rows checked against the photos. The annotations themselves are mine. |
| Claude Opus 5.5 (Claude Code) | Notebook 4 cells 5 (first pipeline test), 7 (passport generator, `report/passports/*.png`, `passport_summary.csv`), 8 (confidence vs anomaly score, attribute overlap), 9 (nearest-training distance, PCA, Spearman), 10 (full-data classifier on the personal photos) | [AI-CODE] ([HUMAN-CHECK] on cells 5 and 7) | Code to run the frozen models on the personal photos, lay out the passports and produce the analysis figures. | Passport values (material, confidence, top-5 attributes, anomaly score) cross-checked against a separate run of the frozen models on single photos (cell 5); figure values compared with `passport_summary.csv`. |
| Claude Opus 5.5 (vision) | Notebook 4 cell 6 (`PASSPORT_TEXT`): description and uncertainty fields of every passport (Figures 30–33) | [AI-VISION] [AI-LANGUAGE] [HUMAN-DATA] [HUMAN-CHECK] | First draft of the short description and uncertainty statement of each passport, written from my photos and the model outputs. | Each text read against its photo, heatmap, mask and scores; wrong or unsupported statements corrected. Photos and annotations are mine. |
| Claude Opus 5.5 (Claude Code) | Notebook 4 cells 32 (recording the frozen Task D responses) and 35 (Task D scoring and confidence plot, Figure 29) | [AI-CODE] | Code to store the 15 raw responses and score them against the answer key. | Stored strings compared with the raw responses in the appendix; the answer key is only read in cell 35, after all responses were frozen; accuracies counted by hand (5/5, 4/5, 4/5). |
| GPT-5.6-sol (2026-10-04) | Task D: the 15 responses (D1–D3 × Q1–Q5), Task D results and Figure 29, appendix | [AI-VISION] [HUMAN-CHECK] | Defect decisions, defect names, confidences and evidence sentences for the five queries. | Responses frozen before opening `solution/task_d_answer_key.csv`, then compared with the answer key and with the query images. |
| Claude Opus 5.5 (Claude Code) | Whole report (`REPORT.md`) | [AI-LANGUAGE] [HUMAN-CHECK] | Proofreading: spelling, grammar and rewording of formulations throughout the report. | Every suggested change reviewed and accepted or rejected; numbers and claims checked against the notebook outputs. |

## Appendix: Task D prompts and raw responses

All 15 queries were run with GPT-5.6-sol on 2026-10-04 in the order Q1 to Q5 for every condition. The responses below are copied verbatim and were frozen before the answer key was opened. The D3 support images are the fixed set from `task_d_support.csv`; they are the `000.png` files of each defect folder, which belong to the public validation split (even index), whereas the queries are `001.png` files from the test split (odd index).

### D1 prompt (identical for all queries)

```
Inspect this surface image. Is anything visibly abnormal? 
Return only valid JSON withis_defective, defect_name, confidence, and evidence.
```

### D3 line appended to every D2 prompt

```
One of the attached image provides examples for each type of defect, use it to establish the defect type.
```

### Q1: carpet

**D2 prompt** (the D3 prompt is the same with the D3 line appended):

```
Inspect this surface image and find if there is among the following :
- color: A localized region has an abnormal colour or tone.
- cut: A sharp incision or sliced region interrupts the texture.
- hole: Material is missing in a compact hole-like region.
- metal_contamination: A metallic foreign object lies on the surface.
- thread: A foreign thread lies across the regular texture.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```

**D3 support images:** one validation example each of color, cut, hole, metal_contamination, thread.

**Raw responses:**

- D1:

  ```json
  {"is_defective":true,"defect_name":"localized weave distortion/damaged fibers","confidence":0.98,"evidence":"A clearly visible abnormal dark, irregular patch appears in the lower-left/center region, where the otherwise regular woven pattern is disrupted and fibers appear bunched, displaced, or damaged."}
  ```

- D2:

  ```json
  {"is_defective":true,"defect_name":"color","confidence":0.96,"evidence":"A localized dark discolored region is visible slightly left of center in the lower-middle portion of the woven surface."}
  ```

- D3:

  ```json
  {"is_defective":true,"defect_name":"color","confidence":0.96,"evidence":"A localized dark-toned patch is visible in the lower-left/central region while the woven texture remains structurally intact, closely matching the provided color-defect example."}
  ```

### Q2: grid

**D2 prompt** (the D3 prompt is the same with the D3 line appended):

```
Inspect this surface image and find if there is among the following :
- bent: The regular grid lines are locally warped or displaced.
- broken: Part of the repeating grid is interrupted or missing.
- glue: A localized glue deposit changes the surface appearance.
- metal_contamination: A metallic foreign object lies on the surface.
- thread: A foreign thread lies across the regular texture.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```

**D3 support images:** one validation example each of bent, broken, glue, metal_contamination, thread.

**Raw responses:**

- D1:

  ```json
  {"is_defective":true,"defect_name":"broken mesh strands","confidence":0.99,"evidence":"Multiple clear discontinuities are visible in the repeating mesh pattern, with several strands broken, bent, and displaced around the upper-center, lower-left, and lower-right regions."}
  ```

- D2:

  ```json
  {"is_defective":true,"defect_name":"broken","confidence":0.99,"evidence":"Multiple grid strands are visibly interrupted with separated, free-ended segments, especially in the upper-middle, lower-left, and lower-right regions."}
  ```

- D3:

  ```json
  {"is_defective":true,"defect_name":"broken","confidence":0.98,"evidence":"Several grid strands are visibly interrupted with separated, free-ended segments, closely matching the provided broken-defect example."}
  ```

### Q3: leather

**D2 prompt** (the D3 prompt is the same with the D3 line appended):

```
Inspect this surface image and find if there is among the following :
- color: A localized region has an abnormal colour or tone.
- cut: A sharp incision or sliced region interrupts the texture.
- fold: The surface is creased or folded out of its normal plane.
- glue: A localized glue deposit changes the surface appearance.
- poke: A small puncture-like indentation or opening is visible.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```

**D3 support images:** one validation example each of color, cut, fold, glue, poke.

**Raw responses:**

- D1:

  ```json
  {"is_defective":true,"defect_name":"puncture/tear","confidence":0.99,"evidence":"A distinct small puncture with a short tear and raised/damaged material is visible slightly left of center, disrupting the otherwise consistent textured surface."}
  ```

- D2:

  ```json
  {"is_defective":true,"defect_name":"poke","confidence":0.97,"evidence":"A small puncture-like opening with localized material disruption is visible slightly left of center."}
  ```

- D3:

  ```json
  {"is_defective":true,"defect_name":"cut","confidence":0.96,"evidence":"A sharp elongated incision with a small opened/damaged endpoint interrupts the leather texture near the lower-left central region, closely matching the provided cut example."}
  ```

### Q4: tile

**D2 prompt** (the D3 prompt is the same with the D3 line appended):

```
Inspect this surface image and find if there is among the following :
- crack: A narrow fracture-like line crosses the material surface.
- glue_strip: An elongated strip of adhesive lies on the tile.
- gray_stroke: A grey linear mark appears across the tile.
- oil: An oily region changes colour or reflectance.
- rough: A localized patch is visibly rougher than normal.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```

**D3 support images:** one validation example each of crack, glue_strip, gray_stroke, oil, rough.

**Raw responses:**

- D1:

  ```json
  {"is_defective":true,"defect_name":"crack","confidence":0.99,"evidence":"A prominent Y-shaped dark crack is visible through the center of the surface, with three distinct branches disrupting the surrounding material."}
  ```

- D2:

  ```json
  {"is_defective":true,"defect_name":"crack","confidence":0.99,"evidence":"A prominent narrow dark fracture branches from the lower center into two diagonal lines extending toward the upper edges."}
  ```

- D3:

  ```json
  {"is_defective":true,"defect_name":"crack","confidence":0.99,"evidence":"Multiple narrow dark fracture-like lines converge near the center and extend across the material surface, closely matching the provided crack example."}
  ```

### Q5: wood

**D2 prompt** (the D3 prompt is the same with the D3 line appended):

```
Inspect this surface image and find if there is among the following :
- color: A localized region has an abnormal colour or tone.
- combined: More than one defect mechanism appears in the same image.
- hole: Material is missing in a compact hole-like region.
- liquid: A liquid stain changes colour or reflectance locally.
- scratch: A thin elongated abrasion crosses the surface.
Only consider a SINGLE label otherwise set defect name to "good".
Return only valid JSON with is_defective, defect_name, confidence, and evidence.
```

**D3 support images:** one validation example each of color, combined, hole, liquid, scratch.

**Raw responses:**

- D1:

  ```json
  {"is_defective":true,"defect_name":"split/crack","confidence":0.98,"evidence":"A prominent vertical separation runs nearly the full height slightly right of center, forming an irregular open split that is distinct from the surrounding wood grain."}
  ```

- D2:

  ```json
  {"is_defective":true,"defect_name":"scratch","confidence":0.98,"evidence":"A prominent thin vertical abrasion interrupts the wood surface slightly right of center, with exposed lighter material along its edges."}
  ```

- D3:

  ```json
  {"is_defective":false,"defect_name":"good","confidence":0.97,"evidence":"The surface shows normal longitudinal wood grain and a natural seam-like boundary, with no localized abnormal color, compact hole, liquid stain, thin abrasion, or combination of the provided defect types."}
  ```

