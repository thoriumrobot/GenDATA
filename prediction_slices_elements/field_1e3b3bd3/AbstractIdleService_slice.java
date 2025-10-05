// Source-based slice around line 44
// Method: com.google.common.util.concurrent.AbstractIdleService.threadNameSupplier

 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@GwtIncompatible
@J2ktIncompatible
public abstract class AbstractIdleService implements Service {

  /* Thread names will look like {@code "MyService STARTING"}. */
  private final Supplier<String> threadNameSupplier = new ThreadNameSupplier();

  @WeakOuter
  private final class ThreadNameSupplier implements Supplier<String> {
    @Override
    public String get() {
      return serviceName() + " " + state();
    }
  }

  /* use AbstractService for state management */
