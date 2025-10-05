// Source-based slice around line 38
// Method: <com.google.common.eventbus.Subscriber: Subscriber create(EventBus,Object,Method)>

 *
 * <p>Two subscribers are equivalent when they refer to the same method on the same object (not
 * class). This property is used to ensure that no subscriber method is registered more than once.
 *
 * @author Colin Decker
 */
class Subscriber {

  /** Creates a {@code Subscriber} for {@code method} on {@code listener}. */
  static Subscriber create(EventBus bus, Object listener, Method method) {
    return isDeclaredThreadSafe(method)
        ? new Subscriber(bus, listener, method)
        : new SynchronizedSubscriber(bus, listener, method);
  }

  /** The event bus this subscriber belongs to. */
  @Weak private final EventBus bus;

  /** The object with the subscriber method. */
  @VisibleForTesting final Object target;
