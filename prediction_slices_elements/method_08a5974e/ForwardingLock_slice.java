// Source-based slice around line 27
// Method: <com.google.common.util.concurrent.ForwardingLock: Lock delegate()>

import com.google.common.annotations.J2ktIncompatible;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
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
