// Source-based slice around line 250
// Method: <com.google.common.primitives.Shorts: short max(short)>

   * Returns the greatest value present in {@code array}.
   *
   * @param array a <i>nonempty</i> array of {@code short} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  @GwtIncompatible(
      "Available in GWT! Annotation is to avoid conflict with GWT specialization of base class.")
  public static short max(short... array) {
    checkArgument(array.length > 0);
    short max = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] > max) {
        max = array[i];
      }
    }
    return max;
  }

