// Source-based slice around line 56
// Method: <com.google.common.collect.RegularImmutableSet: boolean contains(Object)>


  RegularImmutableSet(Object[] elements, int hashCode, @Nullable Object[] table, int mask) {
    this.elements = elements;
    this.hashCode = hashCode;
    this.table = table;
    this.mask = mask;
  }

  @Override
  public boolean contains(@Nullable Object target) {
    @Nullable Object[] table = this.table;
    if (target == null || table.length == 0) {
      return false;
    }
    for (int i = Hashing.smearedHash(target); ; i++) {
      i &= mask;
      Object candidate = table[i];
      if (candidate == null) {
        return false;
      } else if (candidate.equals(target)) {
