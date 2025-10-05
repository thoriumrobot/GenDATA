// Source-based slice around line 56
// Method: <com.google.common.util.concurrent.ForwardingCondition: void signal()>

    return delegate().awaitNanos(nanosTimeout);
  }

  @Override
  public boolean awaitUntil(Date deadline) throws InterruptedException {
    return delegate().awaitUntil(deadline);
  }

  @Override
  public void signal() {
    delegate().signal();
  }

  @Override
  public void signalAll() {
    delegate().signalAll();
  }
}
