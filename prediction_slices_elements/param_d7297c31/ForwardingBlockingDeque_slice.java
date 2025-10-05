// Source-based slice around line 102
// Method: <com.google.common.collect.ForwardingBlockingDeque: E pollLast(long,TimeUnit)>

    return delegate().takeLast();
  }

  @Override
  public @Nullable E pollFirst(long timeout, TimeUnit unit) throws InterruptedException {
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
