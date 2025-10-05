// Source-based slice around line 138
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testEqualsAndHashCodeForwarded()>

                }
                return false;
              }
            };
          }
        },
        "Runnable");
  }

  public void testEqualsAndHashCodeForwarded() {
    tester.includingEquals();
    tester.testForwarding(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {
              @Override
              public boolean equals(@Nullable Object o) {
                if (o instanceof ForwardingRunnable) {
