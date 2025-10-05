// Source-based slice around line 50
// Method: <com.google.common.util.concurrent.ForwardingLock: void unlock()>

    return delegate().tryLock();
  }

  @Override
  public boolean tryLock(long time, TimeUnit unit) throws InterruptedException {
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
