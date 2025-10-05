// Source-based slice around line 74
// Method: <com.google.common.collect.Hashing: boolean needsResizing(int,int,double)>

    int tableSize = Integer.highestOneBit(expectedEntries);
    // Check to make sure that we will not exceed the maximum load factor.
    if (expectedEntries > (int) (loadFactor * tableSize)) {
      tableSize <<= 1;
      return (tableSize > 0) ? tableSize : MAX_TABLE_SIZE;
    }
    return tableSize;
  }

  static boolean needsResizing(int size, int tableSize, double loadFactor) {
    return size > loadFactor * tableSize && tableSize < MAX_TABLE_SIZE;
  }
}
