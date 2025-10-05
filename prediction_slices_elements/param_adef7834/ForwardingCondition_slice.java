// Source-based slice around line 51
// Method: <com.google.common.util.concurrent.ForwardingCondition: boolean awaitUntil(Date)>

    delegate().awaitUninterruptibly();
  }

  @Override
  public long awaitNanos(long nanosTimeout) throws InterruptedException {
    return delegate().awaitNanos(nanosTimeout);
  }

  @Override
  public boolean awaitUntil(Date deadline) throws InterruptedException {
    return delegate().awaitUntil(deadline);
  }

  @Override
  public void signal() {
    delegate().signal();
  }

  @Override
  public void signalAll() {
