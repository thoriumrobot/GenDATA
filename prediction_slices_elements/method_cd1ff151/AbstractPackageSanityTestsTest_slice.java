// Source-based slice around line 77
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_ignoreUnderscores()>

    assertThat(findClassesToTest(ImmutableList.of(PublicFoo.class))).contains(PublicFoo.class);
  }

  public void testFindClassesToTest_ignoreClasses() {
    sanityTests.ignoreClasses(Predicates.<Object>equalTo(PublicFoo.class));
    assertThat(findClassesToTest(ImmutableList.of(PublicFoo.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(Foo.class))).contains(Foo.class);
  }

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
