// Source-based slice around line 59
// Method: com.google.common.io.ReaderInputStream.charBuffer

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
   * draining it is flipped (undrained bytes between position and limit).
   */
  private ByteBuffer byteBuffer;

  /** Whether we've finished reading the reader. */
  private boolean endOfInput;
