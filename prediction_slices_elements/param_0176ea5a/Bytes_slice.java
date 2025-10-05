// Source-based slice around line 148
// Method: <com.google.common.primitives.Bytes: int lastIndexOf(byte[],byte,int,int)>

   * @param target a primitive {@code byte} value
   * @return the greatest index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int lastIndexOf(byte[] array, byte target) {
    return lastIndexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int lastIndexOf(byte[] array, byte target, int start, int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
