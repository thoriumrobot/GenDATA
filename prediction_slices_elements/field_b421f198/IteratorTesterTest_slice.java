// Source-based slice around line 157
// Method: com.google.common.collect.testing.IteratorTesterTest.STEPS

          return new IteratorWithJdkBug6529795<>(iterator);
        }
      }.test();
    } catch (AssertionError e) {
      return;
    }
    fail("Should have caught jdk6 bug in target iterator");
  }

  private static final int STEPS = 3;

  static class TesterThatCountsCalls extends IteratorTester<Integer> {
    TesterThatCountsCalls() {
      super(STEPS, MODIFIABLE, newArrayList(1), IteratorTester.KnownOrder.KNOWN_ORDER);
    }

    int numCallsToNewTargetIterator;
    int numCallsToVerify;

    @Override
