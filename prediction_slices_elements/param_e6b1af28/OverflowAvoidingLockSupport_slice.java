// Source-based slice around line 36
// Method: <com.google.common.util.concurrent.OverflowAvoidingLockSupport: void parkNanos(Object,long)>

 */
@J2ktIncompatible
@GwtIncompatible
final class OverflowAvoidingLockSupport {
  // Represents the max nanoseconds representable on a linux timespec with a 32 bit tv_sec
  static final long MAX_NANOSECONDS_THRESHOLD = (1L + Integer.MAX_VALUE) * 1_000_000_000L - 1L;

  private OverflowAvoidingLockSupport() {}

  static void parkNanos(@Nullable Object blocker, long nanos) {
    // Even in the extremely unlikely event that a thread unblocks itself early after only 68 years,
    // this is indistinguishable from a spurious wakeup, which LockSupport allows.
    LockSupport.parkNanos(blocker, min(nanos, MAX_NANOSECONDS_THRESHOLD));
  }
}
