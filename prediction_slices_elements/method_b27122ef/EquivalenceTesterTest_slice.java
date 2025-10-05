// Source-based slice around line 51
// Method: <com.google.common.testing.EquivalenceTesterTest: void testOf_nullPointerException()>


  @Override
  public void setUp() throws Exception {
    super.setUp();
    this.equivalenceMock = new MockEquivalence();
    this.tester = EquivalenceTester.of(equivalenceMock);
  }

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
