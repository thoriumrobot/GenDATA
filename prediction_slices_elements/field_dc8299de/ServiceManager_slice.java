// Source-based slice around line 126
// Method: com.google.common.util.concurrent.ServiceManager.logger

 * <p>This class uses the ServiceManager's methods to start all of its services, to respond to
 * service failure and to ensure that when the JVM is shutting down all the services are stopped.
 *
 * @author Luke Sandberg
 * @since 14.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class ServiceManager implements ServiceManagerBridge {
  private static final LazyLogger logger = new LazyLogger(ServiceManager.class);
  private static final ListenerCallQueue.Event<Listener> HEALTHY_EVENT =
      new ListenerCallQueue.Event<Listener>() {
        @Override
        public void call(Listener listener) {
          listener.healthy();
        }

        @Override
        public String toString() {
          return "healthy()";
