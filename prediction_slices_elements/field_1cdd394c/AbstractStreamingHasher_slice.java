// Source-based slice around line 33
// Method: com.google.common.hash.AbstractStreamingHasher.buffer

 * A convenience base class for implementors of {@code Hasher}; handles accumulating data until an
 * entire "chunk" (of implementation-dependent length) is ready to be hashed.
 *
 * @author Kevin Bourrillion
 * @author Dimitris Andreou
 */
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
