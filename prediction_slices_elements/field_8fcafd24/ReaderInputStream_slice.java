// Source-based slice around line 53
// Method: com.google.common.io.ReaderInputStream.singleByte

 * controlled, async APIs.
 *
 * @author Chris Nokleberg
 */
@J2ktIncompatible
@GwtIncompatible
final class ReaderInputStream extends InputStream {
  private final Reader reader;
  private final CharsetEncoder encoder;
  private final byte[] singleByte = new byte[1];

  /**
   * charBuffer holds characters that have been read from the Reader but not encoded yet. The buffer
   * is perpetually "flipped" (unencoded characters between position and limit).
   */
  private CharBuffer charBuffer;

  /**
   * byteBuffer holds encoded characters that have not yet been sent to the caller of the input
   * stream. When encoding it is "unflipped" (encoded bytes between 0 and position) and when
