// Source-based slice around line 496
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testCovariantReturn()>

          func, obj);
    }

    @Override
    public String toString() {
      return delegate.toString();
    }
  }

  public void testCovariantReturn() {
    new ForwardingWrapperTester()
        .testForwarding(
            Sub.class,
            new Function<Sub, Sub>() {
              @Override
              public Sub apply(Sub sub) {
                return new ForwardingSub(sub);
              }
            });
  }
