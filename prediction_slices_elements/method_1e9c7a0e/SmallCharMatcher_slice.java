// Source-based slice around line 136
// Method: <com.google.common.base.SmallCharMatcher: void setBits(BitSet)>

      } else { // Linear probing.
        index = (index + 1) & mask;
      }
      // Check to see if we wrapped around the whole table.
    } while (index != startingIndex);
    return false;
  }

  @Override
  void setBits(BitSet table) {
    if (containsZero) {
      table.set(0);
    }
    for (char c : this.table) {
      if (c != 0) {
        table.set(c);
      }
    }
  }
}
