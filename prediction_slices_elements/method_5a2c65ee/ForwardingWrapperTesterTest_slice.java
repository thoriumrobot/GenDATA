// Source-based slice around line 70
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testVoidMethodForwarding()>

        ParameterTypesDifferent.class,
        new Function<ParameterTypesDifferent, ParameterTypesDifferent>() {
          @Override
          public ParameterTypesDifferent apply(ParameterTypesDifferent delegate) {
            return new ParameterTypesDifferentForwarder(delegate);
          }
        });
  }

  public void testVoidMethodForwarding() {
    tester.testForwarding(
        Runnable.class,
        new Function<Runnable, Runnable>() {
          @Override
          public Runnable apply(Runnable runnable) {
            return new ForwardingRunnable(runnable);
          }
        });
  }

