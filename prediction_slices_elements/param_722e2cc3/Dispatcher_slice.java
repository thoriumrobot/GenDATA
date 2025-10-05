// Source-based slice around line 71
// Method: <com.google.common.eventbus.Dispatcher: void dispatch(Object,Iterator)>

   * Returns a dispatcher that dispatches events to subscribers immediately as they're posted
   * without using an intermediate queue to change the dispatch order. This is effectively a
   * depth-first dispatch order, vs. breadth-first when using a queue.
   */
  static Dispatcher immediate() {
    return ImmediateDispatcher.INSTANCE;
  }

  /** Dispatches the given {@code event} to the given {@code subscribers}. */
  abstract void dispatch(Object event, Iterator<Subscriber> subscribers);

  /** Implementation of a {@link #perThreadDispatchQueue()} dispatcher. */
  private static final class PerThreadQueuedDispatcher extends Dispatcher {

    // This dispatcher matches the original dispatch behavior of EventBus.

    /** Per-thread queue of events to dispatch. */
    @SuppressWarnings("ThreadLocalUsage") // Each Dispatcher needs its own state.
    private final ThreadLocal<Queue<Event>> queue =
        new ThreadLocal<Queue<Event>>() {
