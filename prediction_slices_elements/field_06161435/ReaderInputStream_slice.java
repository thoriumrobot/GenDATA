// Source-based slice around line 66
// Method: com.google.common.io.ReaderInputStream.byteBuffer

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

  /** Whether we're copying encoded bytes to the caller's buffer. */
  private boolean draining;

  /** Whether we've successfully flushed the encoder. */
  private boolean doneFlushing;

