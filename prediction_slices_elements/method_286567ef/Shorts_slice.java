// Source-based slice around line 229
// Method: <com.google.common.primitives.Shorts: short min(short)>

   * Returns the least value present in {@code array}.
   *
   * @param array a <i>nonempty</i> array of {@code short} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  @GwtIncompatible(
      "Available in GWT! Annotation is to avoid conflict with GWT specialization of base class.")
  public static short min(short... array) {
    checkArgument(array.length > 0);
    short min = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] < min) {
        min = array[i];
      }
    }
    return min;
  }

