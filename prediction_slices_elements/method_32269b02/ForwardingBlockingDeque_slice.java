// Source-based slice around line 95
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: E pollFirst(long,TimeUnit)>

    return delegate().takeFirst();
  }

  @Override
  public E takeLast() throws InterruptedException {
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
