// Source-based slice around line 29
// Method: com.google.common.eventbus.SubscriberExceptionContext.subscriber


/**
 * Context for an exception thrown by a subscriber.
 *
 * @since 16.0
 */
public class SubscriberExceptionContext {
  private final EventBus eventBus;
  private final Object event;
  private final Object subscriber;
  private final Method subscriberMethod;

  /**
   * @param eventBus The {@link EventBus} that handled the event and the subscriber. Useful for
   *     broadcasting a new event based on the error.
   * @param event The event object that caused the subscriber to throw.
   * @param subscriber The source subscriber context.
   * @param subscriberMethod the subscribed method.
   */
  SubscriberExceptionContext(
