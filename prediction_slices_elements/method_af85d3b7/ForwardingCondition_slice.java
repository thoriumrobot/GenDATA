// Source-based slice around line 31
// Method: <com.google.common.util.concurrent.ForwardingCondition: void await()>


/** Forwarding wrapper around a {@code Condition}. */
@SuppressWarnings("WaitNotInLoop") // We are just delegating; _our user_ must loop.
@J2ktIncompatible
@GwtIncompatible
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
