// Source-based slice around line 320
// Method: <com.google.common.primitives.UnsignedInts: int decode(String)>

   *   <li>{@code 0X}<i>HexDigits</i>
   *   <li>{@code #}<i>HexDigits</i>
   *   <li>{@code 0}<i>OctalDigits</i>
   * </ul>
   *
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code int} value
   * @since 13.0
   */
  @CanIgnoreReturnValue
  public static int decode(String stringValue) {
    ParseRequest request = ParseRequest.fromString(stringValue);

    try {
      return parseUnsignedInt(request.rawValue, request.radix);
    } catch (NumberFormatException e) {
      NumberFormatException decodeException =
          new NumberFormatException("Error parsing value: " + stringValue);
      decodeException.initCause(e);
      throw decodeException;
    }
