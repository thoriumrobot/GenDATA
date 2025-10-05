// Source-based slice around line 62
// Method: com.google.common.util.concurrent.ListenerCallQueue.listeners

 * #dispatch} is expected to be called concurrently, it is idempotent.
 */
@J2ktIncompatible
@GwtIncompatible
final class ListenerCallQueue<L> {
  // TODO(cpovirk): consider using the logger associated with listener.getClass().
  private static final LazyLogger logger = new LazyLogger(ListenerCallQueue.class);

  // TODO(chrisn): promote AppendOnlyCollection for use here.
  private final List<PerListenerQueue<L>> listeners =
      Collections.synchronizedList(new ArrayList<PerListenerQueue<L>>());

  /** Method reference-compatible listener event. */
  interface Event<L> {
    /** Call a method on the listener. */
    void call(L listener);
  }

  /**
   * Adds a listener that will be called using the given executor when events are later {@link
