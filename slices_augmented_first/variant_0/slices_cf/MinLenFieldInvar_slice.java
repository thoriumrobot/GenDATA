    @Positive
    public InvalidSub() {
    @Positive
      super(new int[] {1, 2});
    @Positive
    }
    @Positive
  }

    @Positive
  class ValidSub extends Super {
    @Positive
    public final int[] validSubField;

    @Positive
    public ValidSub(int[] validSubField) {
    @Positive
      super(new int[] {1, 2, 3, 4});
    @Positive
      this.validSubField = validSubField;
    @Positive
    }
