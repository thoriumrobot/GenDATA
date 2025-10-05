// Source-based slice around line 105
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: void put(E)>

    return delegate().pollFirst(timeout, unit);
  }

  @Override
  public @Nullable E pollLast(long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().pollLast(timeout, unit);
  }

  @Override
  public void put(E e) throws InterruptedException {
    delegate().put(e);
  }

  @Override
  public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().offer(e, timeout, unit);
  }

  @Override
  public E take() throws InterruptedException {
