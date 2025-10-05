// Source-based slice around line 256
// Method: <com.google.common.primitives.Ints: int max(int)>

   * Returns the greatest value present in {@code array}.
   *
   * @param array a <i>nonempty</i> array of {@code int} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  @GwtIncompatible(
      "Available in GWT! Annotation is to avoid conflict with GWT specialization of base class.")
  public static int max(int... array) {
    checkArgument(array.length > 0);
    int max = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] > max) {
        max = array[i];
      }
    }
    return max;
  }

