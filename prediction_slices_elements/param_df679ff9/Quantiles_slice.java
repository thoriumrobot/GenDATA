// Source-based slice around line 665
// Method: <com.google.common.math.Quantiles: int chooseNextSelection(int[],int,int,int,int)>

   * Chooses the next selection to do from the required selections. It is required that the array
   * {@code allRequired} is sorted and that {@code allRequired[i]} are in the range [{@code from},
   * {@code to}] for all {@code i} in the range [{@code requiredFrom}, {@code requiredTo}]. The
   * value returned by this method is the {@code i} in that range such that {@code allRequired[i]}
   * is as close as possible to the center of the range [{@code from}, {@code to}]. Choosing the
   * value closest to the center of the range first is the most efficient strategy because it
   * minimizes the size of the subranges from which the remaining selections must be done.
   */
  private static int chooseNextSelection(
      int[] allRequired, int requiredFrom, int requiredTo, int from, int to) {
    if (requiredFrom == requiredTo) {
      return requiredFrom; // only one thing to choose, so choose it
    }

    // Find the center and round down. The true center is either centerFloor or halfway between
    // centerFloor and centerFloor + 1.
    int centerFloor = (from + to) >>> 1;

    // Do a binary search until we're down to the range of two which encloses centerFloor (unless
    // all values are lower or higher than centerFloor, in which case we find the two highest or
