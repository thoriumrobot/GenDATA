// Source-based slice around line 65
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: void putFirst(E)>

  @Override
  protected abstract BlockingDeque<E> delegate();

  @Override
  public int remainingCapacity() {
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
