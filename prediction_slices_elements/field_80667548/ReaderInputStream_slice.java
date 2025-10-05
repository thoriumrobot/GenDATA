// Source-based slice around line 72
// Method: com.google.common.io.ReaderInputStream.draining

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

  /**
   * Creates a new input stream that will encode the characters from {@code reader} into bytes using
   * the given character set. Malformed input and unmappable characters will be replaced.
   *
   * @param reader input source
   * @param charset character set used for encoding chars to bytes
