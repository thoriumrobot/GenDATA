// Source-based slice around line 201
// Method: <com.google.common.primitives.Bytes: byte[] ensureCapacity(byte[],int,int)>

   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
   * @param minLength the minimum length the returned array must guarantee
   * @param padding an extra amount to "grow" the array by if growth is necessary
   * @throws IllegalArgumentException if {@code minLength} or {@code padding} is negative
   * @return an array containing the values of {@code array}, with guaranteed minimum length {@code
   *     minLength}
   */
  public static byte[] ensureCapacity(byte[] array, int minLength, int padding) {
    checkArgument(minLength >= 0, "Invalid minLength: %s", minLength);
    checkArgument(padding >= 0, "Invalid padding: %s", padding);
    return (array.length < minLength) ? Arrays.copyOf(array, minLength + padding) : array;
  }

  /**
   * Returns an array containing each value of {@code collection}, converted to a {@code byte} value
   * in the manner of {@link Number#byteValue}.
   *
   * <p>Elements are copied from the argument collection as if by {@code collection.toArray()}.
