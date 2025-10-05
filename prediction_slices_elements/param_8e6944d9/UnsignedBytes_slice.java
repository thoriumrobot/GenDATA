// Source-based slice around line 249
// Method: <com.google.common.primitives.UnsignedBytes: String join(String,byte)>

  /**
   * Returns a string containing the supplied {@code byte} values separated by {@code separator}.
   * For example, {@code join(":", (byte) 1, (byte) 2, (byte) 255)} returns the string {@code
   * "1:2:255"}.
   *
   * @param separator the text that should appear between consecutive values in the resulting string
   *     (but not at the start or end)
   * @param array an array of {@code byte} values, possibly empty
   */
  public static String join(String separator, byte... array) {
    checkNotNull(separator);
    if (array.length == 0) {
      return "";
    }

    // For pre-sizing a builder, just get the right order of magnitude
    StringBuilder builder = new StringBuilder(array.length * (3 + separator.length()));
    builder.append(toUnsignedInt(array[0]));
    for (int i = 1; i < array.length; i++) {
      builder.append(separator).append(toString(array[i]));
