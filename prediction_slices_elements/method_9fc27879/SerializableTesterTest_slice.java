// Source-based slice around line 123
// Method: <com.google.common.testing.SerializableTesterTest: void assertContains(String,String)>

      }

      @Override
      public int hashCode() {
        return 1;
      }
    }
  }

  private static void assertContains(String expectedSubstring, String actual) {
    // TODO(kevinb): use a Truth assertion here
    if (!actual.contains(expectedSubstring)) {
      fail("expected <" + actual + "> to contain <" + expectedSubstring + ">");
    }
  }
}
