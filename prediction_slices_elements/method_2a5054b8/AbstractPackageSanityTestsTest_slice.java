// Source-based slice around line 92
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_withCorrespondingTestClassAndExplicitlyTested()>

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
  }

  public void testFindClassesToTest_withCorrespondingTestClass_noTestName() {
    assertThat(findClassesToTest(ImmutableList.of(Foo.class, FooTest.class)))
        .containsExactly(Foo.class);
  }

