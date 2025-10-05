// Source-based slice around line 158
// Method: com.google.common.eventbus.EventBus.executor

 *
 * @author Cliff Biffle
 * @since 10.0
 */
public class EventBus {

  private static final Logger logger = Logger.getLogger(EventBus.class.getName());

  private final String identifier;
  private final Executor executor;
  private final SubscriberExceptionHandler exceptionHandler;

  private final SubscriberRegistry subscribers = new SubscriberRegistry(this);
  private final Dispatcher dispatcher;

  /** Creates a new EventBus named "default". */
  public EventBus() {
    this("default");
  }

