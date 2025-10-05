// Source-based slice around line 46
// Method: <com.google.common.util.concurrent.ForwardingCondition: long awaitNanos(long)>

    return delegate().await(time, unit);
  }

  @Override
  public void awaitUninterruptibly() {
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
