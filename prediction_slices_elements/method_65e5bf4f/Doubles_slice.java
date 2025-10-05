// Source-based slice around line 348
// Method: <com.google.common.primitives.Doubles: double[] ensureCapacity(double[],int,int)>

   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
   * @param minLength the minimum length the returned array must guarantee
   * @param padding an extra amount to "grow" the array by if growth is necessary
   * @throws IllegalArgumentException if {@code minLength} or {@code padding} is negative
   * @return an array containing the values of {@code array}, with guaranteed minimum length {@code
   *     minLength}
   */
  public static double[] ensureCapacity(double[] array, int minLength, int padding) {
    checkArgument(minLength >= 0, "Invalid minLength: %s", minLength);
    checkArgument(padding >= 0, "Invalid padding: %s", padding);
    return (array.length < minLength) ? Arrays.copyOf(array, minLength + padding) : array;
  }

  /**
   * Returns a string containing the supplied {@code double} values, converted to strings as
   * specified by {@link Double#toString(double)}, and separated by {@code separator}. For example,
   * {@code join("-", 1.0, 2.0, 3.0)} returns the string {@code "1.0-2.0-3.0"}.
   *
