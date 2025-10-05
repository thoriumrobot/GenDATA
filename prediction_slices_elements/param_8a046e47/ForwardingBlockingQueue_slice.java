// Source-based slice around line 65
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: boolean offer(E,long,TimeUnit)>


  @CanIgnoreReturnValue
  @Override
  public int drainTo(Collection<? super E> c) {
    return delegate().drainTo(c);
  }

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
