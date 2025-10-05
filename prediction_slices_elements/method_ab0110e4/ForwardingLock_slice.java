// Source-based slice around line 35
// Method: <com.google.common.util.concurrent.ForwardingLock: void lockInterruptibly()>

abstract class ForwardingLock implements Lock {
  abstract Lock delegate();

  @Override
  public void lock() {
    delegate().lock();
  }

  @Override
  public void lockInterruptibly() throws InterruptedException {
    delegate().lockInterruptibly();
  }

  @Override
  public boolean tryLock() {
    return delegate().tryLock();
  }

  @Override
  public boolean tryLock(long time, TimeUnit unit) throws InterruptedException {
