// Source-based slice around line 166
// Method: com.google.common.util.concurrent.AbstractService.isStopped

      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(RUNNING) >= 0;
    }
  }

  private final Guard isStopped = new IsStoppedGuard();

  @WeakOuter
  private final class IsStoppedGuard extends Guard {
    IsStoppedGuard() {
      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(TERMINATED) >= 0;
