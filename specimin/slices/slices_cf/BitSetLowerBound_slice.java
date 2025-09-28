  private void m(BitSet b) {
    b.set(b.nextClearBit(0));
    // next set bit does not have to exist
    // :: error: (argument)
    b.clear(b.nextSetBit(0));
    @GTENegativeOne int i = b.nextSetBit(0);

    @GTENegativeOne int j = b.previousClearBit(-1);
    @GTENegativeOne int k = b.previousSetBit(-1);
  }
