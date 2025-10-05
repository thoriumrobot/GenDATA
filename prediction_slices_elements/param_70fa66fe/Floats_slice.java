// Source-based slice around line 230
// Method: <com.google.common.primitives.Floats: float max(float)>

   * {@link Math#max(float, float)}.
   *
   * @param array a <i>nonempty</i> array of {@code float} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  @GwtIncompatible(
      "Available in GWT! Annotation is to avoid conflict with GWT specialization of base class.")
  public static float max(float... array) {
    checkArgument(array.length > 0);
    float max = array[0];
    for (int i = 1; i < array.length; i++) {
      max = Math.max(max, array[i]);
    }
    return max;
  }

  /**
   * Returns the value nearest to {@code value} which is within the closed range {@code [min..max]}.
