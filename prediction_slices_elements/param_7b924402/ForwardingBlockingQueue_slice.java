// Source-based slice around line 76
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: void put(E)>

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
    return delegate().remainingCapacity();
  }

  @CanIgnoreReturnValue // TODO(kak): consider removing this
  @Override
