// Source-based slice around line 210
// Method: <com.google.common.primitives.Floats: float min(float)>

   * Math#min(float, float)}.
   *
   * @param array a <i>nonempty</i> array of {@code float} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  @GwtIncompatible(
      "Available in GWT! Annotation is to avoid conflict with GWT specialization of base class.")
  public static float min(float... array) {
    checkArgument(array.length > 0);
    float min = array[0];
    for (int i = 1; i < array.length; i++) {
      min = Math.min(min, array[i]);
    }
    return min;
  }

  /**
   * Returns the greatest value present in {@code array}, using the same rules of comparison as
