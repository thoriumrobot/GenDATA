// Source-based slice around line 39
// Method: com.google.common.hash.AbstractStreamingHasher.chunkSize

// TODO(kevinb): this class still needs some design-and-document-for-inheritance love
abstract class AbstractStreamingHasher extends AbstractHasher {
  /** Buffer via which we pass data to the hash algorithm (the implementor) */
  private final ByteBuffer buffer;

  /** Number of bytes to be filled before process() invocation(s). */
  private final int bufferSize;

  /** Number of bytes processed per process() invocation. */
  private final int chunkSize;

  /**
   * Constructor for use by subclasses. This hasher instance will process chunks of the specified
   * size.
   *
   * @param chunkSize the number of bytes available per {@link #process(ByteBuffer)} invocation;
   *     must be at least 4
   */
  protected AbstractStreamingHasher(int chunkSize) {
    this(chunkSize, chunkSize);
