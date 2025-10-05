// Source-based slice around line 81
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testToStringForwarding()>

        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable);
          }
        });
  }

  public void testToStringForwarding() {
    tester.testForwarding(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {
              @Override
              public String toString() {
                return runnable.toString();
              }
