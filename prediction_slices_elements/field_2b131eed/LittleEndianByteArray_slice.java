// Source-based slice around line 46
// Method: com.google.common.hash.LittleEndianByteArray.byteArray

 * @author Kevin Damm
 * @author Kyle Maddison
 */
final class LittleEndianByteArray {

  /**
   * The instance that actually does the work; delegates to VarHandle, Unsafe, or a Java-8
   * compatible pure-Java fallback.
   */
  private static final LittleEndianBytes byteArray = makeGetter();

  /**
   * Load 8 bytes into long in a little endian manner, from the substring between position and
   * position + 8. The array must have at least 8 bytes from offset (inclusive).
   *
   * @param input the input bytes
   * @param offset the offset into the array at which to start
   * @return a long of a concatenated 8 bytes
   */
  static long load64(byte[] input, int offset) {
