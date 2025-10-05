// Source-based slice around line 248
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testUnexpectedException()>

              public boolean hasNext() {
                return false;
              }
            };
          }
        };
    assertFailure(tester);
  }

  public void testUnexpectedException() {
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            1, MODIFIABLE, newArrayList(1), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return new ThrowingIterator<>(new IllegalStateException());
          }
        };
    assertFailure(tester);
  }
