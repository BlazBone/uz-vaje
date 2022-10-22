### Why would you use different color maps?

- color values are impertredet different, for excample if we plot
- how the image is vied

### How is inverting a grayscale value defined for uint8?

- 255 - value

### The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?

- we cannot compare histograms that are not normalized (pictures can have different number of pixels)
- histogram should show the procentage of pixels that have certain value when comparing

### Based on the results, which order of erosion and dilation operations produces opening and which closing?

- Opening = Erosion then dialation
- Closing = Dialation then erosion

### Why is the background included in the mask and not the object? How would you fix that in general? (just inverting the mask if necessary doesn’t count)

- because the background is lighter than the object itself
- inverting image? almost the same as inverting the mask
- user has option to choose which one he would like to pick
