// Source-based slice around line 271
// Method: <com.google.common.primitives.Booleans: boolean[] ensureCapacity(boolean[],int,int)>

   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
   * @param minLength the minimum length the returned array must guarantee
   * @param padding an extra amount to "grow" the array by if growth is necessary
   * @throws IllegalArgumentException if {@code minLength} or {@code padding} is negative
   * @return an array containing the values of {@code array}, with guaranteed minimum length {@code
   *     minLength}
   */
  public static boolean[] ensureCapacity(boolean[] array, int minLength, int padding) {
    checkArgument(minLength >= 0, "Invalid minLength: %s", minLength);
    checkArgument(padding >= 0, "Invalid padding: %s", padding);
    return (array.length < minLength) ? Arrays.copyOf(array, minLength + padding) : array;
  }

  /**
   * Returns a string containing the supplied {@code boolean} values separated by {@code separator}.
   * For example, {@code join("-", false, true, false)} returns the string {@code
   * "false-true-false"}.
   *
