// Source-based slice around line 399
// Method: <com.google.common.primitives.Ints: Converter stringConverter()>

   * Integer#decode} and {@link Integer#toString()}. The returned converter throws {@link
   * NumberFormatException} if the input string is invalid.
   *
   * <p><b>Warning:</b> please see {@link Integer#decode} to understand exactly how strings are
   * parsed. For example, the string {@code "0123"} is treated as <i>octal</i> and converted to the
   * value {@code 83}.
   *
   * @since 16.0
   */
  public static Converter<String, Integer> stringConverter() {
    return IntConverter.INSTANCE;
  }

  /**
   * Returns an array containing the same values as {@code array}, but guaranteed to be of a
   * specified minimum length. If {@code array} already has a length of at least {@code minLength},
   * it is returned directly. Otherwise, a new array of size {@code minLength + padding} is
   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
