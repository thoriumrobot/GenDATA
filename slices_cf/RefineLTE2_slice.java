  public void add(int elt) {
    if (num_values == values.length) {
      values = null;
      // :: error: (unary.increment)
      num_values++;
      return;
    }
    values[num_values] = elt;
    num_values++;
  }
