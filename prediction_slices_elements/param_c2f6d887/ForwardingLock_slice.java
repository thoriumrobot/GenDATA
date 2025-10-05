// Source-based slice around line 45
// Method: <com.google.common.util.concurrent.ForwardingLock: boolean tryLock(long,TimeUnit)>

    delegate().lockInterruptibly();
  }

  @Override
  public boolean tryLock() {
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
