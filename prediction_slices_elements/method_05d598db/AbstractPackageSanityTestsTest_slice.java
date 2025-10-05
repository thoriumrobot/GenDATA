// Source-based slice around line 98
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_withCorrespondingTestClass_noTestName()>

        .containsExactly(Foo.class);
  }

  public void testFindClassesToTest_withCorrespondingTestClassAndExplicitlyTested() {
    ImmutableList<Class<?>> classes = ImmutableList.of(Foo.class, FooTest.class);
    assertThat(findClassesToTest(classes, "testPublic")).isEmpty();
    assertThat(findClassesToTest(classes, "testNotThere", "testPublic")).isEmpty();
  }

  public void testFindClassesToTest_withCorrespondingTestClass_noTestName() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, FooTest.class)))
        .containsExactly(Foo.class);
  }

  static class EmptyTestCase {}

  static class EmptyTest {}

  static class EmptyTests {}

