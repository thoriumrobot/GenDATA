// Source-based slice around line 127
// Method: <com.google.common.collect.ForwardingBlockingDeque: int drainTo(Collection)>

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
    return delegate().drainTo(c, maxElements);
  }
}
