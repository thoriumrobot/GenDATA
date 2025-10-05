// Source-based slice around line 82
// Method: <com.google.common.collect.ForwardingBlockingDeque: boolean offerLast(E,long,TimeUnit)>

    delegate().putLast(e);
  }

  @Override
  public boolean offerFirst(E e, long timeout, TimeUnit unit) throws InterruptedException {
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
