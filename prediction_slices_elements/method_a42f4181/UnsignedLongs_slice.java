// Source-based slice around line 382
// Method: <com.google.common.primitives.UnsignedLongs: long decode(String)>

   *   <li>{@code #}<i>HexDigits</i>
   *   <li>{@code 0}<i>OctalDigits</i>
   * </ul>
   *
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code long}
   *     value
   * @since 13.0
   */
  @CanIgnoreReturnValue
  public static long decode(String stringValue) {
    ParseRequest request = ParseRequest.fromString(stringValue);

    try {
      return parseUnsignedLong(request.rawValue, request.radix);
    } catch (NumberFormatException e) {
      NumberFormatException decodeException =
          new NumberFormatException("Error parsing value: " + stringValue);
      decodeException.initCause(e);
      throw decodeException;
    }
