// Source-based slice around line 213
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableList()>

    int[] a1 = generator.generateFresh(int[].class);
    int[] a2 = generator.generateFresh(int[].class);
    assertTrue(a1[0] != a2[0]);
  }

  public void testRange() {
    assertFreshInstance(new TypeToken<Range<String>>() {});
  }

  public void testImmutableList() {
    assertFreshInstance(new TypeToken<ImmutableList<String>>() {});
  }

  public void testImmutableSet() {
    assertFreshInstance(new TypeToken<ImmutableSet<String>>() {});
  }

  public void testImmutableSortedSet() {
    assertFreshInstance(new TypeToken<ImmutableSortedSet<String>>() {});
  }
