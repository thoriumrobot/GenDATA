// Source-based slice around line 85
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_withCorrespondingTestClassButNotExplicitlyTested()>


  public void testFindClassesToTest_ignoreUnderscores() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, Foo_Bar.class)))
        .containsExactly(Foo.class, Foo_Bar.class);
    sanityTests.ignoreClasses(AbstractPackageSanityTests.UNDERSCORE_IN_NAME);
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, Foo_Bar.class)))
        .containsExactly(Foo.class);
  }

  public void testFindClassesToTest_withCorrespondingTestClassButNotExplicitlyTested() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, FooTest.class), "testNotThere"))
        .containsExactly(Foo.class);
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, FooTest.class), "testNotPublic"))
        .containsExactly(Foo.class);
  }

  public void testFindClassesToTest_withCorrespondingTestClassAndExplicitlyTested() {
    ImmutableList<Class<?>> classes = ImmutableList.of(Foo.class, FooTest.class);
    assertThat(findClassesToTest(classes, "testPublic")).isEmpty();
    assertThat(findClassesToTest(classes, "testNotThere", "testPublic")).isEmpty();
