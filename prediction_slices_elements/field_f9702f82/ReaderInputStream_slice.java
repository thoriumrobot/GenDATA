// Source-based slice around line 75
// Method: com.google.common.io.ReaderInputStream.doneFlushing

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
   * @param bufferSize size of internal input and output buffers
   * @throws IllegalArgumentException if bufferSize is non-positive
   */
