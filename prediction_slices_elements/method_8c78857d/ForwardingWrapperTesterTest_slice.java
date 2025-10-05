// Source-based slice around line 114
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForwardHashCode()>

              public String toString() {
                return "";
              }
            };
          }
        },
        "toString()");
  }

  public void testFailsToForwardHashCode() {
    tester.includingEquals();
    assertFailure(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable) {

              @SuppressWarnings("EqualsHashCode")
              @Override
