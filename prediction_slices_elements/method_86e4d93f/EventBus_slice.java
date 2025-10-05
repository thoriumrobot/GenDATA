// Source-based slice around line 211
// Method: <com.google.common.eventbus.EventBus: Executor executor()>

   * Returns the identifier for this event bus.
   *
   * @since 19.0
   */
  public final String identifier() {
    return identifier;
  }

  /** Returns the default executor this event bus uses for dispatching events to subscribers. */
  final Executor executor() {
    return executor;
  }

  /** Handles the given exception thrown by a subscriber with the given context. */
  void handleSubscriberException(Throwable e, SubscriberExceptionContext context) {
    checkNotNull(e);
    checkNotNull(context);
    try {
      exceptionHandler.handleException(e, context);
    } catch (Throwable e2) {
