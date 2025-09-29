    // :: error: (array.length.negative)
    int[] b = new int[Integer.valueOf(x)];
  }

  @PolyValue int poly(@PolyValue int y) {
    return y;
  }
}
