// Source-based slice around line 75
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: boolean offerFirst(E,long,TimeUnit)>

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
    return delegate().offerLast(e, timeout, unit);
  }

  @Override
  public E takeFirst() throws InterruptedException {
