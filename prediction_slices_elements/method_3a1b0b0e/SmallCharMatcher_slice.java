// Source-based slice around line 112
// Method: <com.google.common.base.SmallCharMatcher: boolean matches(char)>

        }
        // Linear probing.
        index = (index + 1) & mask;
      }
    }
    return new SmallCharMatcher(table, filter, containsZero, description);
  }

  @Override
  public boolean matches(char c) {
    if (c == 0) {
      return containsZero;
    }
    if (!checkFilter(c)) {
      return false;
    }
    int mask = table.length - 1;
    int startingIndex = smear(c) & mask;
    int index = startingIndex;
    do {
