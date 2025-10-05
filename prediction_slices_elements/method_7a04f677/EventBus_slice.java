// Source-based slice around line 216
// Method: <com.google.common.eventbus.EventBus: void handleSubscriberException(Throwable,SubscriberExceptionContext)>

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
      // if the handler threw an exception... well, just log it
      logger.log(
          Level.SEVERE,
          String.format(Locale.ROOT, "Exception %s thrown while handling exception: %s", e2, e),
          e2);
