// Source-based slice around line 55
// Method: com.google.common.util.concurrent.AbstractIdleService.delegate

  @WeakOuter
  private final class ThreadNameSupplier implements Supplier<String> {
    @Override
    public String get() {
      return serviceName() + " " + state();
    }
  }

  /* use AbstractService for state management */
  private final Service delegate = new DelegateService();

  @WeakOuter
  private final class DelegateService extends AbstractService {
    @Override
    protected final void doStart() {
      renamingDecorator(executor(), threadNameSupplier)
          .execute(
              () -> {
                try {
                  startUp();
