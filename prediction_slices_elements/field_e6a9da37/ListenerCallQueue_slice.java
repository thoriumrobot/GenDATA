// Source-based slice around line 59
// Method: com.google.common.util.concurrent.ListenerCallQueue.logger

 * {@link #enqueue} and {@link #dispatch} are 2 different methods. It is expected that the decision
 * to run a particular event is made during the state change, but the decision to actually invoke
 * the listeners can be delayed slightly so that locks can be dropped. Also, because {@link
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
