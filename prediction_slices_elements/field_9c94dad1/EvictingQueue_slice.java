// Source-based slice around line 50
// Method: com.google.common.collect.EvictingQueue.delegate

 *
 * <p>This class is not thread-safe, and does not accept null elements.
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
