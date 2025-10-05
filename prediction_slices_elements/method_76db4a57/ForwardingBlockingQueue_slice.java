// Source-based slice around line 59
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: int drainTo(Collection)>


  @CanIgnoreReturnValue
  @Override
  public int drainTo(Collection<? super E> c, int maxElements) {
    return delegate().drainTo(c, maxElements);
  }

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
