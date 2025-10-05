// Source-based slice around line 268
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testNotInterfaceType()>

          @Override
          public Adder apply(Adder adder) {
            return new FailsToPropagateException(adder);
          }
        },
        "add(",
        "exception");
  }

  public void testNotInterfaceType() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new ForwardingWrapperTester()
                .testForwarding(String.class, Functions.<String>identity()));
  }

  public void testNulls() {
    new NullPointerTester()
        .setDefault(Class.class, Runnable.class)
