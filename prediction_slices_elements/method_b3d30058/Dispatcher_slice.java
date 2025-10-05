// Source-based slice around line 47
// Method: <com.google.common.eventbus.Dispatcher: Dispatcher perThreadDispatchQueue()>

   * Returns a dispatcher that queues events that are posted reentrantly on a thread that is already
   * dispatching an event, guaranteeing that all events posted on a single thread are dispatched to
   * all subscribers in the order they are posted.
   *
   * <p>When all subscribers are dispatched to using a <i>direct</i> executor (which dispatches on
   * the same thread that posts the event), this yields a breadth-first dispatch order on each
   * thread. That is, all subscribers to a single event A will be called before any subscribers to
   * any events B and C that are posted to the event bus by the subscribers to A.
   */
  static Dispatcher perThreadDispatchQueue() {
    return new PerThreadQueuedDispatcher();
  }

  /**
   * Returns a dispatcher that queues events that are posted in a single global queue. This behavior
   * matches the original behavior of AsyncEventBus exactly, but is otherwise not especially useful.
   * For async dispatch, an {@linkplain #immediate() immediate} dispatcher should generally be
   * preferable.
   */
  static Dispatcher legacyAsync() {
