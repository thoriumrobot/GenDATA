// Source-based slice around line 41
// Method: <com.google.common.util.concurrent.ForwardingCondition: void awaitUninterruptibly()>

    delegate().await();
  }

  @Override
  public boolean await(long time, TimeUnit unit) throws InterruptedException {
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
