// Source-based slice around line 97
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForwardToString()>

              @Override
              public String toString() {
                return runnable.toString();
              }
            };
          }
        });
  }

  public void testFailsToForwardToString() {
    assertFailure(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {
              @Override
              public String toString() {
                return "";
              }
