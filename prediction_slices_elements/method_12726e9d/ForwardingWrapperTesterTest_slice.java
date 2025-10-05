// Source-based slice around line 230
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testForwardsToTheWrongMethod()>

          @Override
          public Adder apply(Adder adder) {
            return new FailsToForwardParameters(adder);
          }
        },
        "add(",
        "Parameter #0");
  }

  public void testForwardsToTheWrongMethod() {
    assertFailure(
        Arithmetic.class,
        new Function<Arithmetic, Arithmetic>() {
          @Override
          public Arithmetic apply(Arithmetic adder) {
            return new ForwardsToTheWrongMethod(adder);
          }
        },
        "minus");
  }
