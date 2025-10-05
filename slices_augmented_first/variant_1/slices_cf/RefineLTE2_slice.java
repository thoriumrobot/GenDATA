    @Positive
  protected int @MinLen(1) [] values;

    @Positive
  @LTEqLengthOf("values") int num_values;

    @Positive
  public void add(int elt) {
    @Positive
    if (num_values == values.length) {
    @Positive
      values = null;
      // :: error: (unary.increment)
    @Positive
      num_values++;
    @Positive
      return;
    @Positive
    }
    @Positive
    values[num_values] = elt;
