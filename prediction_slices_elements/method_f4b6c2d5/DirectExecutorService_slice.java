// Source-based slice around line 90
// Method: <com.google.common.util.concurrent.DirectExecutorService: boolean awaitTermination(long,TimeUnit)>


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
        if (shutdown && runningTasks == 0) {
          return true;
        } else if (nanos <= 0) {
          return false;
        } else {
          long now = System.nanoTime();
          NANOSECONDS.timedWait(lock, nanos);
