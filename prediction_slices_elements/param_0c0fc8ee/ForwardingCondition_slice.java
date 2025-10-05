// Source-based slice around line 36
// Method: <com.google.common.util.concurrent.ForwardingCondition: boolean await(long,TimeUnit)>

abstract class ForwardingCondition implements Condition {
  abstract Condition delegate();

  @Override
  public void await() throws InterruptedException {
    delegate().await();
  }

  @Override
  public boolean await(long time, TimeUnit unit) throws InterruptedException {
    return delegate().await(time, unit);
  }

  @Override
  public void awaitUninterruptibly() {
    delegate().awaitUninterruptibly();
  }

  @Override
  public long awaitNanos(long nanosTimeout) throws InterruptedException {
