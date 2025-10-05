// Source-based slice around line 129
// Method: <com.google.common.testing.EqualsTesterTest: void testInvalidEqualsNull()>

      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, obj + " must be Object#equals to itself");
      return;
    }
    fail("Should get non-reflexive error");
  }

  /** Test proper handling where an object tests equal to null */
  public void testInvalidEqualsNull() {
    Object obj = new InvalidEqualsNullObject();
    equalsTester.addEqualityGroup(obj);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, obj + " must not be Object#equals to null");
      return;
    }
    fail("Should get equal to null error");
  }
