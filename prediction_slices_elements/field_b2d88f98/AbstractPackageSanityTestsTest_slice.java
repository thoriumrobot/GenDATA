// Source-based slice around line 50
// Method: com.google.common.testing.AbstractPackageSanityTestsTest.sanityTests

   *
   * We'd just use PackageSanityTests directly, saving us from needing this separate type, but we're
   * currently skipping MediumTests on Android, and we skip them by not making them present at
   * runtime at all. I could just make _this_ test a MediumTest, but then it wouldn't run on
   * Android.... The right long-term fix is probably to get MediumTests running under Android by
   * default and then suppress them strategically as needed.
   */
  public static final class ConcretePackageSanityTests extends AbstractPackageSanityTests {}

  private final AbstractPackageSanityTests sanityTests = new ConcretePackageSanityTests();

  public void testFindClassesToTest_testClass() {
    assertThat(findClassesToTest(ImmutableList.of(EmptyTest.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTests.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTestCase.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTestSuite.class))).isEmpty();
  }

  public void testFindClassesToTest_noCorrespondingTestClass() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class))).containsExactly(Foo.class);
