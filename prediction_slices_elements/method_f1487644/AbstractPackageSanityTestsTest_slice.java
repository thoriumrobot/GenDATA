// Source-based slice around line 132
// Method: <com.google.common.testing.AbstractPackageSanityTestsTest: List findClassesToTest(Iterable,String)>

    void testNotPublic() {}
  }

  // Shouldn't be mistaken as Foo's test
  static class Foo2Test {
    @SuppressWarnings("unused") // accessed reflectively
    public void testPublic() {}
  }

  private List<Class<?>> findClassesToTest(
      Iterable<? extends Class<?>> classes, String... explicitTestNames) {
    return sanityTests.findClassesToTest(classes, Arrays.asList(explicitTestNames));
  }
}
