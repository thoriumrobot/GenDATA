// Source-based slice around line 59
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_noCorrespondingTestClass()>

  private final AbstractPackageSanityTests sanityTests = new ConcretePackageSanityTests();

  public void testFindClassesToTest_testClass() {
    assertThat(findClassesToTest(ImmutableList.of(EmptyTest.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTests.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTestCase.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(EmptyTestSuite.class))).isEmpty();
  }

  public void testFindClassesToTest_noCorrespondingTestClass() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class))).containsExactly(Foo.class);
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, Foo2Test.class)))
        .containsExactly(Foo.class);
  }

  public void testFindClassesToTest_publicApiOnly() {
    sanityTests.publicApiOnly();
    assertThat(findClassesToTest(ImmutableList.of(Foo.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(PublicFoo.class))).contains(PublicFoo.class);
  }
