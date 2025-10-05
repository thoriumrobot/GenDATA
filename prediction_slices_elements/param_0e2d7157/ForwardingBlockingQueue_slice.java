// Source-based slice around line 71
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: E poll(long,TimeUnit)>


  @CanIgnoreReturnValue // TODO(kak): consider removing this
  @Override
  public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().offer(e, timeout, unit);
  }

  @CanIgnoreReturnValue // TODO(kak): consider removing this
  @Override
  public @Nullable E poll(long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().poll(timeout, unit);
  }

  @Override
  public void put(E e) throws InterruptedException {
    delegate().put(e);
  }

  @Override
  public int remainingCapacity() {
