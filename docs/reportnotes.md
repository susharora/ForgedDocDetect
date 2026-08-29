**RN1:**
Yes, a model trained exclusively on synthetic data generated via 3x3 homography matrices will likely suffer a significant performance drop when exposed to real-world optical distortion.
Because the training pipeline enforces strict collinearity, the model's neural network weights overfit to the mathematical perfection of straight lines and rigid perspective geometry. When presented with physical bending or barrel distortion from a smartphone lens, the model encounters spatial patterns that exist entirely outside its learned distribution.

This geometric bias manifests in several ways during model evaluation:

* **Localization Failure:** The network will attempt to force a straight-line fit onto the curved edges of the document. The predicted bounding polygons will invariably clip the bowed corners of the ID card or include excessive background pixels along the inward-curving edges.
* **Feature Misalignment:** If the localization model feeds its cropped output to a downstream OCR engine or fraud detection classifier, the rigid 3x3 unwarping process will fail to flatten the curve. The resulting canonical image will remain warped, misaligning the text fields and degrading the accuracy of subsequent extraction layers.
* **Adversarial Vulnerability:** This rigid training bias introduces a structural weakness. When evaluating the robustness of models trained on synthetic identity datasets like FantasyID or FREUID, this blind spot can be directly exploited. Attackers can intentionally apply subtle, non-linear spatial transformations—such as localized elastic warping or grid distortions—to forged documents. Because the defense model has only learned to recognize and reverse linear homographies, these non-linear spatial attacks can bypass forgery detection mechanisms while remaining visually imperceptible to humans.

To build a robust model capable of handling real-world captures, the synthetic generation pipeline must move beyond simple affine matrices. The training data must actively inject non-linear augmentations—such as radial distortion maps, random mesh deformations, or Thin-Plate Spline bending—to force the model to learn the physical imperfections of actual camera lenses and bent cards.

