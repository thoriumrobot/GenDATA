// Source-based slice around line 56
// Method: com.google.common.util.concurrent.AbstractService.STARTING_EVENT

 * AbstractExecutionThreadService} if you need only a single execution thread.
 *
 * @author Jesse Wilson
 * @author Luke Sandberg
 * @since 1.0
 */
@GwtIncompatible
@J2ktIncompatible
public abstract class AbstractService implements Service {
  private static final ListenerCallQueue.Event<Listener> STARTING_EVENT =
      new ListenerCallQueue.Event<Listener>() {
        @Override
        public void call(Listener listener) {
          listener.starting();
        }

        @Override
        public String toString() {
          return "starting()";
        }
