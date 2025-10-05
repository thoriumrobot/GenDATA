// Source-based slice around line 147
// Method: <com.google.common.testing.EquivalenceTesterTest: void testTest_inequivalence()>

          .hasMessageThat()
          .contains(
              "TestObject{group=1, item=2} [group 1, item 2] must be equivalent to "
                  + "TestObject{group=1, item=3} [group 1, item 3]");
      return;
    }
    fail();
  }

  public void testTest_inequivalence() {
    Object group1Item1 = new TestObject(1, 1);
    Object group2Item1 = new TestObject(2, 1);

    equivalenceMock.expectEquivalent(group1Item1, group2Item1);
    equivalenceMock.expectDistinct(group2Item1, group1Item1);

    equivalenceMock.expectHash(group1Item1, 1);
    equivalenceMock.expectHash(group2Item1, 2);

    equivalenceMock.replay();
