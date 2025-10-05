// Source-based slice around line 52
// Method: com.google.common.collect.EvictingQueue.maxSize

 *
 * @author Kurt Alfred Kluever
 * @since 15.0
 */
@GwtCompatible
public final class EvictingQueue<E> extends ForwardingQueue<E> implements Serializable {

  private final Queue<E> delegate;

  @VisibleForTesting final int maxSize;

  private EvictingQueue(int maxSize) {
    checkArgument(maxSize >= 0, "maxSize (%s) must >= 0", maxSize);
    this.delegate = new ArrayDeque<>(maxSize);
    this.maxSize = maxSize;
  }

  /**
   * Creates and returns a new evicting queue that will hold up to {@code maxSize} elements.
   *
