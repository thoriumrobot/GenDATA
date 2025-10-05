// Source-based slice around line 92
// Method: <com.google.common.collect.ForwardingBlockingDeque: E takeLast()>

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
    return delegate().pollFirst(timeout, unit);
  }

  @Override
  public @Nullable E pollLast(long timeout, TimeUnit unit) throws InterruptedException {
