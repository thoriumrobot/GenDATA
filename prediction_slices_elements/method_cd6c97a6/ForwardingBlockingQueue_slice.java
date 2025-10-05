// Source-based slice around line 81
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: int remainingCapacity()>

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
  public E take() throws InterruptedException {
    return delegate().take();
  }
}
