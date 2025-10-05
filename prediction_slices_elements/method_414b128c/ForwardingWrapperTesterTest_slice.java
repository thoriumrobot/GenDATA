// Source-based slice around line 164
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForwardEquals()>

              @Override
              public int hashCode() {
                return runnable.hashCode();
              }
            };
          }
        });
  }

  public void testFailsToForwardEquals() {
    tester.includingEquals();
    assertFailure(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {
              @Override
              public int hashCode() {
                return runnable.hashCode();
