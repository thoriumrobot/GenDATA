// Source-based slice around line 87
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: E take()>

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
