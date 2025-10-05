// Source-based slice around line 230
// Method: <com.google.common.primitives.UnsignedBytes: byte parseUnsignedByte(String,int)>

   * @param radix the radix to use while parsing {@code string}
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code byte} with
   *     the given radix, or if {@code radix} is not between {@link Character#MIN_RADIX} and {@link
   *     Character#MAX_RADIX}.
   * @throws NullPointerException if {@code string} is null (in contrast to {@link
   *     Byte#parseByte(String)})
   * @since 13.0
   */
  @CanIgnoreReturnValue
  public static byte parseUnsignedByte(String string, int radix) {
    int parse = Integer.parseInt(checkNotNull(string), radix);
    // We need to throw a NumberFormatException, so we have to duplicate checkedCast. =(
    if (parse >> Byte.SIZE == 0) {
      return (byte) parse;
    } else {
      throw new NumberFormatException("out of range: " + parse);
    }
  }

  /**
