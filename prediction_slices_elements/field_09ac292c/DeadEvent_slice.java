// Source-based slice around line 32
// Method: com.google.common.eventbus.DeadEvent.source

 *
 * <p>Registering a DeadEvent subscriber is useful for debugging or logging, as it can detect
 * misconfigurations in a system's event distribution.
 *
 * @author Cliff Biffle
 * @since 10.0
 */
public class DeadEvent {

  private final Object source;
  private final Object event;

  /**
   * Creates a new DeadEvent.
   *
   * @param source object broadcasting the DeadEvent (generally the {@link EventBus}).
   * @param event the event that could not be delivered.
   */
  public DeadEvent(Object source, Object event) {
    this.source = checkNotNull(source);
