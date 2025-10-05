// Source-based slice around line 30
// Method: <com.google.common.util.concurrent.ForwardingLock: void lock()>

import java.util.concurrent.locks.Lock;

/** Forwarding wrapper around a {@code Lock}. */
@J2ktIncompatible
@GwtIncompatible
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
