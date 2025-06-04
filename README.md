
You can use the same conda env in main branch.

This repository contains code to train an image reconstruction model using
`google/siglip2-so400m-patch16-512` paired with
`Efficient-Large-Model/Sana_1600M_512px_diffusers`.

1. **Training**

   * Open `train_image_reconstruction.py` and update the [data path](https://github.com/JiuhaiChen/BLIP3o/blob/Reconstruction/train_image_reconstruction.py#L360) to point at your input `.tar` dataset (the script currently only accepts tar‐packed image files).
   * Run the `run.sh` to train the reconstruction model.

2. **Inference**

   * After training completes, change the `inference.py` to specify:

     * The path to your trained model checkpoint.
     * The original image file you wish to reconstruct.

