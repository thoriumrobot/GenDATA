// Source-based slice around line 120
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: E poll(long,TimeUnit)>

    return delegate().offer(e, timeout, unit);
  }

  @Override
  public E take() throws InterruptedException {
    return delegate().take();
  }

  @Override
  public @Nullable E poll(long timeout, TimeUnit unit) throws InterruptedException {
    return delegate().poll(timeout, unit);
  }

  @Override
  public int drainTo(Collection<? super E> c) {
    return delegate().drainTo(c);
  }

  @Override
  public int drainTo(Collection<? super E> c, int maxElements) {
