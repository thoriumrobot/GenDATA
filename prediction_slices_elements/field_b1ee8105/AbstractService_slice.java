// Source-based slice around line 138
// Method: com.google.common.util.concurrent.AbstractService.isStoppable

      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state() == NEW;
    }
  }

  private final Guard isStoppable = new IsStoppableGuard();

  @WeakOuter
  private final class IsStoppableGuard extends Guard {
    IsStoppableGuard() {
      super(AbstractService.this.monitor);
    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(RUNNING) <= 0;
