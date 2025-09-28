      int @MinLen(2) [] b = a;
    }
  }

  void neq_zero_special_case(int[] a) {
    if (a.length != 0) {
      int @MinLen(1) [] b = a;
    }
  }
}
