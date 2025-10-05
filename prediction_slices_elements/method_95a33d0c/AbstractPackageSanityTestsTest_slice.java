// Source-based slice around line 65
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: void testFindClassesToTest_publicApiOnly()>

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

  public void testFindClassesToTest_ignoreClasses() {
    sanityTests.ignoreClasses(Predicates.<Object>equalTo(PublicFoo.class));
    assertThat(findClassesToTest(ImmutableList.of(PublicFoo.class))).isEmpty();
    assertThat(findClassesToTest(ImmutableList.of(Foo.class))).contains(Foo.class);
  }
