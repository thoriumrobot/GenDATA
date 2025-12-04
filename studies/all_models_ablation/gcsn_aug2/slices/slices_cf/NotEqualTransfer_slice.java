    }
    @Positive
  }

    @Positive
  void neq_bad_check(int[] a) {
    @Positive
    if (1 != a.length) {
    @Positive
      int x = 1; // do nothing.
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
