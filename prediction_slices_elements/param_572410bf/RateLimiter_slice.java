// Source-based slice around line 493
// Method: <com.google.common.util.concurrent.RateLimiter: void checkPermits(int)>

        protected void sleepMicrosUninterruptibly(long micros) {
          if (micros > 0) {
            Uninterruptibles.sleepUninterruptibly(micros, MICROSECONDS);
          }
        }
      };
    }
  }

  private static void checkPermits(int permits) {
    checkArgument(permits > 0, "Requested permits (%s) must be positive", permits);
  }
}
