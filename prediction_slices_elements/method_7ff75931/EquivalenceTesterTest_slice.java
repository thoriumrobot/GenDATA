// Source-based slice around line 116
// Method: <com.google.common.testing.EquivalenceTesterTest: void testTest_transitive()>

          .hasMessageThat()
          .contains(
              "TestObject{group=1, item=2} [group 1, item 2] must be equivalent to "
                  + "TestObject{group=1, item=1} [group 1, item 1]");
      return;
    }
    fail();
  }

  public void testTest_transitive() {
    Object group1Item1 = new TestObject(1, 1);
    Object group1Item2 = new TestObject(1, 2);
    Object group1Item3 = new TestObject(1, 3);

    equivalenceMock.expectEquivalent(group1Item1, group1Item2);
    equivalenceMock.expectEquivalent(group1Item1, group1Item3);
    equivalenceMock.expectEquivalent(group1Item2, group1Item1);
    equivalenceMock.expectDistinct(group1Item2, group1Item3);
    equivalenceMock.expectEquivalent(group1Item3, group1Item1);
    equivalenceMock.expectEquivalent(group1Item3, group1Item2);
