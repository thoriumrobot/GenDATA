// Source-based slice around line 618
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testChainingCalls()>

      return delegate.nonChainingCall();
    }

    @Override
    public String toString() {
      return delegate.toString();
    }
  }

  public void testChainingCalls() {
    tester.testForwarding(
        ChainingCalls.class,
        new Function<ChainingCalls, ChainingCalls>() {
          @Override
          public ChainingCalls apply(ChainingCalls delegate) {
            return new ForwardingChainingCalls(delegate);
          }
        });
  }
}
