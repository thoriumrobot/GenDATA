// Source-based slice around line 55
// Method: <com.google.common.util.concurrent.ForwardingLock: Condition newCondition()>

    return delegate().tryLock(time, unit);
  }

  @Override
  public void unlock() {
    delegate().unlock();
  }

  @Override
  public Condition newCondition() {
    return delegate().newCondition();
  }
}
