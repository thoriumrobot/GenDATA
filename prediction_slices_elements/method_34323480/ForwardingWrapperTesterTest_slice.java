// Source-based slice around line 182
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForward()>

              public int hashCode() {
                return runnable.hashCode();
              }
            };
          }
        },
        "Runnable");
  }

  public void testFailsToForward() {
    assertFailure(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {
              @Override
              public void run() {}
            };
          }
