// Source-based slice around line 110
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: boolean offer(E,long,TimeUnit)>

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
    return delegate().take();
  }

  @Override
  public @Nullable E poll(long timeout, TimeUnit unit) throws InterruptedException {
