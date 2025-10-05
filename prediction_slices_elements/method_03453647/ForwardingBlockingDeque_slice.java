// Source-based slice around line 87
// Method: <com.google.common.collect.ForwardingBlockingDeque: E takeFirst()>

    return delegate().offerFirst(e, timeout, unit);
  }

  @Override
  public boolean offerLast(E e, long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().offerLast(e, timeout, unit);
  }

  @Override
  public E takeFirst() throws InterruptedException {
    return delegate().takeFirst();
  }

  @Override
  public E takeLast() throws InterruptedException {
    return delegate().takeLast();
  }

  @Override
  public @Nullable E pollFirst(long timeout, TimeUnit unit) throws InterruptedException {
