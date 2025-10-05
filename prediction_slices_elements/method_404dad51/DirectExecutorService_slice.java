// Source-based slice around line 83
// Method: <com.google.common.util.concurrent.DirectExecutorService: boolean isTerminated()>


  // See newDirectExecutorService javadoc for unusual behavior of this method.
  @Override
  public List<Runnable> shutdownNow() {
    shutdown();
    return ImmutableList.of();
  }

  @Override
  public boolean isTerminated() {
    synchronized (lock) {
      return shutdown && runningTasks == 0;
    }
  }

  @Override
  public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
    long nanos = unit.toNanos(timeout);
    synchronized (lock) {
      while (true) {
