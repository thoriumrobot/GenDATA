// Source-based slice around line 152
// Method: com.google.common.util.concurrent.AbstractService.hasReachedRunning

      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(RUNNING) <= 0;
    }
  }

  private final Guard hasReachedRunning = new HasReachedRunningGuard();

  @WeakOuter
  private final class HasReachedRunningGuard extends Guard {
    HasReachedRunningGuard() {
      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(RUNNING) >= 0;
