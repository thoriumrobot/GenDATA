  void badParam(
      // :: error: (different.length.sequences.offsets)
      @LTLengthOf(
              value = {"a", "b"},
              offset = {"c"})
          int x) {}

  void badParam2(
      // :: error: (different.length.sequences.offsets)
      @LTLengthOf(
              value = {"a"},
              offset = {"c", "d"})
          int x) {}
