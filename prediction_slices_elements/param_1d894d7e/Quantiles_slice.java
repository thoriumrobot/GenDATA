// Source-based slice around line 628
// Method: <com.google.common.math.Quantiles: void selectAllInPlace(int[],int,int,double[],int,int)>

    // The postcondition now holds. So the median, our chosen pivot, is at from.
  }

  /**
   * Performs an in-place selection, like {@link #selectInPlace}, to select all the indexes {@code
   * allRequired[i]} for {@code i} in the range [{@code requiredFrom}, {@code requiredTo}]. These
   * indexes must be sorted in the array and must all be in the range [{@code from}, {@code to}].
   */
  private static void selectAllInPlace(
      int[] allRequired, int requiredFrom, int requiredTo, double[] array, int from, int to) {
    // Choose the first selection to do...
    int requiredChosen = chooseNextSelection(allRequired, requiredFrom, requiredTo, from, to);
    int required = allRequired[requiredChosen];

    // ...do the first selection...
    selectInPlace(required, array, from, to);

    // ...then recursively perform the selections in the range below...
    int requiredBelow = requiredChosen - 1;
    while (requiredBelow >= requiredFrom && allRequired[requiredBelow] == required) {
