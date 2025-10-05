// Source-based slice around line 91
// Method: <com.google.common.testing.EquivalenceTesterTest: void testTest_symmetric()>


    equivalenceMock.replay();

    tester
        .addEquivalenceGroup(group1Item1, group1Item2)
        .addEquivalenceGroup(group2Item1, group2Item2)
        .test();
  }

  public void testTest_symmetric() {
    Object group1Item1 = new TestObject(1, 1);
    Object group1Item2 = new TestObject(1, 2);

    equivalenceMock.expectEquivalent(group1Item1, group1Item2);
    equivalenceMock.expectDistinct(group1Item2, group1Item1);

    equivalenceMock.expectHash(group1Item1, 1);
    equivalenceMock.expectHash(group1Item2, 1);

    equivalenceMock.replay();
