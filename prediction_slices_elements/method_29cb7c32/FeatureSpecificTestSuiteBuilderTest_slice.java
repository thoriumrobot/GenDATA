// Source-based slice around line 40
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilderTest: void testLifecycle()>

  private static final class MyTestSuiteBuilder
      extends FeatureSpecificTestSuiteBuilder<MyTestSuiteBuilder, String> {
    @SuppressWarnings("rawtypes") // class literals
    @Override
    protected List<Class<? extends AbstractTester>> getTesters() {
      return Collections.<Class<? extends AbstractTester>>singletonList(MyTester.class);
    }
  }

  public void testLifecycle() {
    boolean[] setUp = {false};
    Runnable setUpRunnable =
        new Runnable() {
          @Override
          public void run() {
            setUp[0] = true;
          }
        };

    boolean[] tearDown = {false};
