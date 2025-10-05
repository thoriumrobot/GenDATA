// Source-based slice around line 61
// Method: <com.google.common.util.concurrent.ForwardingCondition: void signalAll()>

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
