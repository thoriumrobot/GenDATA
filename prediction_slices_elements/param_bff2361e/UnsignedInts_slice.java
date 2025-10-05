// Source-based slice around line 361
// Method: <com.google.common.primitives.UnsignedInts: int parseUnsignedInt(String,int)>

   * @param string the string containing the unsigned integer representation to be parsed.
   * @param radix the radix to use while parsing {@code s}; must be between {@link
   *     Character#MIN_RADIX} and {@link Character#MAX_RADIX}.
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code int}, or
   *     if supplied radix is invalid.
   * @throws NullPointerException if {@code s} is null (in contrast to {@link
   *     Integer#parseInt(String)})
   */
  @CanIgnoreReturnValue
  public static int parseUnsignedInt(String string, int radix) {
    checkNotNull(string);
    long result = Long.parseLong(string, radix);
    if ((result & INT_MASK) != result) {
      throw new NumberFormatException(
          "Input " + string + " in base " + radix + " is not in the range of an unsigned integer");
    }
    return (int) result;
  }

  /**
