// Source-based slice around line 242
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForwardReturnValue()>

        new Function<Arithmetic, Arithmetic>() {
          @Override
          public Arithmetic apply(Arithmetic adder) {
            return new ForwardsToTheWrongMethod(adder);
          }
        },
        "minus");
  }

  public void testFailsToForwardReturnValue() {
    assertFailure(
        Adder.class,
        new Function<Adder, Adder>() {
          @Override
          public Adder apply(Adder adder) {
            return new FailsToForwardReturnValue(adder);
          }
        },
        "add(",
        "Return value");
