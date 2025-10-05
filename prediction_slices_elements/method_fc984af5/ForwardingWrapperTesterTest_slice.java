// Source-based slice around line 255
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToPropagateException()>

          @Override
          public Adder apply(Adder adder) {
            return new FailsToForwardReturnValue(adder);
          }
        },
        "add(",
        "Return value");
  }

  public void testFailsToPropagateException() {
    assertFailure(
        Adder.class,
        new Function<Adder, Adder>() {
          @Override
          public Adder apply(Adder adder) {
            return new FailsToPropagateException(adder);
          }
        },
        "add(",
        "exception");
