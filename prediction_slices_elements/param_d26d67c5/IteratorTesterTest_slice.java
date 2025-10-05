// Source-based slice around line 322
// Method: <com.google.common.collect.testing.IteratorTesterTest: void assertFailure(IteratorTester)>

              public boolean hasNext() {
                return false;
              }
            };
          }
        };
    assertFailure(tester);
  }

  private static void assertFailure(IteratorTester<?> tester) {
    try {
      tester.test();
    } catch (AssertionError expected) {
      return;
    }
    fail();
  }

  private static final class ThrowingIterator<E> implements Iterator<E> {
    private final RuntimeException ex;
