// Source-based slice around line 198
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testRedundantForwarding()>

              @Override
              public void run() {}
            };
          }
        },
        "run()",
        "Failed to forward");
  }

  public void testRedundantForwarding() {
    assertFailure(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new Runnable() {
              @Override
              public void run() {
                runnable.run();
                runnable.run();
