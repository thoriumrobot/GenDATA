// Source-based slice around line 70
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: void putLast(E)>

    return delegate().remainingCapacity();
  }

  @Override
  public void putFirst(E e) throws InterruptedException {
    delegate().putFirst(e);
  }

  @Override
  public void putLast(E e) throws InterruptedException {
    delegate().putLast(e);
  }

  @Override
  public boolean offerFirst(E e, long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().offerFirst(e, timeout, unit);
  }

  @Override
  public boolean offerLast(E e, long timeout, TimeUnit unit) throws InterruptedException {
