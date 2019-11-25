# In progress:
 *[ ] Replacing SSE with RMSE: Normalised and physical meaning

# Problems
## High priority

## Low priority
*[ ] Improve speed.
    * look at Profiler.
*[ ] In eeo.py, don't work with globals
    * Temporary fix: globals are reset at using eeo.initialization
*[ ] eeo.update_neighbors_priority_rgb:
    * Work with MSE instead of Sum of squared error (and replace/rescale all corresponding values).
    * Should solve interpretability for e.g. thresh_uncertainty as it makes it independent to patchsize.
    
# Solved

 *[x] When inpainting, is it needed that a whole square is inpainted/replaced by the patch, instead of only inpainting the to-be-inpainted pixels?
    * Solve by a boolean flag.
    * Might cause artifacts: "local optimum" that causes the use of the same patch over and over.
