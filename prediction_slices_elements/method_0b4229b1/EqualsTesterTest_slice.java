// Source-based slice around line 194
// Method: <com.google.common.testing.EqualsTesterTest: void testNullEqualityGroup()>

              + " [group 1, item 1] must be equal to the Object#hashCode ("
              + b.hashCode()
              + ") of "
              + b);
      return;
    }
    fail("Should get invalid hashCode error");
  }

  public void testNullEqualityGroup() {
    EqualsTester tester = new EqualsTester();
    assertThrows(NullPointerException.class, () -> tester.addEqualityGroup((Object[]) null));
  }

  public void testNullObjectInEqualityGroup() {
    EqualsTester tester = new EqualsTester();
    NullPointerException e =
        assertThrows(NullPointerException.class, () -> tester.addEqualityGroup(1, null, 3));
    assertErrorMessage(e, "at index 1");
  }
