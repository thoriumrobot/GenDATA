// Source-based slice around line 217
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testFailsToForwardParameters()>

                runnable.run();
              }
            };
          }
        },
        "run()",
        "invoked more than once");
  }

  public void testFailsToForwardParameters() {
    assertFailure(
        Adder.class,
        new Function<Adder, Adder>() {
          @Override
          public Adder apply(Adder adder) {
            return new FailsToForwardParameters(adder);
          }
        },
        "add(",
        "Parameter #0");
