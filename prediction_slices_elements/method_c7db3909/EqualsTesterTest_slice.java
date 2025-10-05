// Source-based slice around line 101
// Method: <com.google.common.testing.EqualsTesterTest: void testTestEqualsEmptyLists()>

      return;
    }
    fail("Should get not equal to equal object error");
  }

  /**
   * Test EqualsTester with no equals or not equals objects. This checks proper handling of null,
   * incompatible class and reflexive tests
   */
  public void testTestEqualsEmptyLists() {
    equalsTester.addEqualityGroup(reference);
    equalsTester.testEquals();
  }

  /**
   * Test EqualsTester after populating equalObjects. This checks proper handling of equality and
   * verifies hashCode for valid objects
   */
  public void testTestEqualsEqualsObjects() {
    equalsTester.addEqualityGroup(reference, equalObject1, equalObject2);
