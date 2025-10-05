// Source-based slice around line 59
// Method: <com.google.common.testing.EquivalenceTesterTest: void testTest()>

  /** Test null reference yields error */
  public void testOf_nullPointerException() {
    assertThrows(NullPointerException.class, () -> EquivalenceTester.of(null));
  }

  public void testTest_noData() {
    tester.test();
  }

  public void testTest() {
    Object group1Item1 = new TestObject(1, 1);
    Object group1Item2 = new TestObject(1, 2);
    Object group2Item1 = new TestObject(2, 1);
    Object group2Item2 = new TestObject(2, 2);

    equivalenceMock.expectEquivalent(group1Item1, group1Item2);
    equivalenceMock.expectDistinct(group1Item1, group2Item1);
    equivalenceMock.expectDistinct(group1Item1, group2Item2);
    equivalenceMock.expectEquivalent(group1Item2, group1Item1);
    equivalenceMock.expectDistinct(group1Item2, group2Item1);
