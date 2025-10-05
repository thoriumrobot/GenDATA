// Source-based slice around line 172
// Method: <com.google.common.testing.EquivalenceTesterTest: void testTest_hash()>

          .hasMessageThat()
          .contains(
              "TestObject{group=1, item=1} [group 1, item 1] must not be equivalent to "
                  + "TestObject{group=2, item=1} [group 2, item 1]");
      return;
    }
    fail();
  }

  public void testTest_hash() {
    Object group1Item1 = new TestObject(1, 1);
    Object group1Item2 = new TestObject(1, 2);

    equivalenceMock.expectEquivalent(group1Item1, group1Item2);
    equivalenceMock.expectEquivalent(group1Item2, group1Item1);

    equivalenceMock.expectHash(group1Item1, 1);
    equivalenceMock.expectHash(group1Item2, 2);

    equivalenceMock.replay();
