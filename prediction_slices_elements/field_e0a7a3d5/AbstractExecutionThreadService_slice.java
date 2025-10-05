// Source-based slice around line 41
// Method: com.google.common.util.concurrent.AbstractExecutionThreadService.delegate

 * if you would like to manage any threading manually.
 *
 * @author Jesse Wilson
 * @since 1.0
 */
@GwtIncompatible
@J2ktIncompatible
public abstract class AbstractExecutionThreadService implements Service {
  /* use AbstractService for state management */
  private final Service delegate =
      new AbstractService() {
        @Override
        protected final void doStart() {
          Executor executor = renamingDecorator(executor(), () -> serviceName());
          executor.execute(
              () -> {
                try {
                  startUp();
                  notifyStarted();
                  // If stopAsync() is called while starting we may be in the STOPPING state in
