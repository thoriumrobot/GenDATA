// Source-based slice around line 57
// Method: <com.google.common.eventbus.Dispatcher: Dispatcher legacyAsync()>

    return new PerThreadQueuedDispatcher();
  }

  /**
   * Returns a dispatcher that queues events that are posted in a single global queue. This behavior
   * matches the original behavior of AsyncEventBus exactly, but is otherwise not especially useful.
   * For async dispatch, an {@linkplain #immediate() immediate} dispatcher should generally be
   * preferable.
   */
  static Dispatcher legacyAsync() {
    return new LegacyAsyncDispatcher();
  }

  /**
   * Returns a dispatcher that dispatches events to subscribers immediately as they're posted
   * without using an intermediate queue to change the dispatch order. This is effectively a
   * depth-first dispatch order, vs. breadth-first when using a queue.
   */
  static Dispatcher immediate() {
    return ImmediateDispatcher.INSTANCE;
