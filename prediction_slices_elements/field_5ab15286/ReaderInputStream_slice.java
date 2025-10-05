// Source-based slice around line 51
// Method: com.google.common.io.ReaderInputStream.reader

 * which is necessarily blocking. By implementing an {@code InputStream} it allows consumers to
 * "pull" as much data as they can handle, which is more convenient when dealing with flow
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
