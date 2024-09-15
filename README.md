# Reference-only-for-Flux
Just code snippet

# Instructions for Using This Repository

- **VRAM Usage**: This method requires increased VRAM usage. The exact amount depends on the image resolution.

## Setup
1. Update three .py files in the comfy/ldm/flux directory.
   **Note**: Make sure to backup the original files before making any changes.
2. Place the `variables.json` file in the root directory of the repository.

## Usage
3. Set the mode in `variables.json` to "write" and use the write workflow.
4. In the workflow:
   - Add a reference image to the "Load Image" node.
   - Add a black image of the same size as the reference image to the "Load Image (as mask)" node.
5. Queue the workflow. (This will create a "tensor data" folder containing latent images with added noise based on the reference image)

6. Change the mode in `variables.json` to "ref" and use the ref workflow.
7. Add a reference image to the "Load Image" node. (It doesn't have to be the same reference image, but it should be the same size)
8. Input your prompt. (Accurate prompting is crucial. I personally use ChatGPT to describe the reference image in 150 words)
9. Queue the workflow.

## variables.json Explanation
- `mode`: Switch between "write" and "ref"
- `kfactor`: Reference strength (1.1-1.3 is recommended. Small changes can significantly affect the result)
- `vfactor`: Another reference strength (can be changed, but not recommended)
- `tfactor_pairs`: Reference strength coefficient for each timestep
- Other parameters are for internal processing and should not be changed.

## Additional Notes
- Image-to-image (i2i) is possible, but you need to run the write mode when changing the number of steps or denoise amount for the first time.
- Comment: This method often struggles with images where the subject is small relative to the background.
