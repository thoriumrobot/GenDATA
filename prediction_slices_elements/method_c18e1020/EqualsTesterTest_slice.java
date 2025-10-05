// Source-based slice around line 156
// Method: <com.google.common.testing.EqualsTesterTest: void testInvalidNotEqualsEqualObject()>

    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e, obj + " must not be Object#equals to an arbitrary object of another class");
      return;
    }
    fail("Should get equal to incompatible class error");
  }

  /** Test proper handling where an object is not equal to one the user has said should be equal */
  public void testInvalidNotEqualsEqualObject() {
    equalsTester.addEqualityGroup(reference, notEqualObject1);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, reference + " [group 1, item 1]");
      assertErrorMessage(e, notEqualObject1 + " [group 1, item 2]");
      return;
    }
    fail("Should get not equal to equal object error");
  }
