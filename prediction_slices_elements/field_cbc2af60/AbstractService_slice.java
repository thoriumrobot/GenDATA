// Source-based slice around line 124
// Method: com.google.common.util.concurrent.AbstractService.isStartable

      @Override
      public String toString() {
        return "stopping({from = " + from + "})";
      }
    };
  }

  private final Monitor monitor = new Monitor();

  private final Guard isStartable = new IsStartableGuard();

  @WeakOuter
  private final class IsStartableGuard extends Guard {
    IsStartableGuard() {
      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state() == NEW;
