    @Positive
  public boolean client2() {
    @Positive
    return withcondpostconditionsfunc2();
    @Positive
  }

    @Positive
  })
    @Positive
  public void client3() {
    @Positive
    withpostconditionfunc1();
    @Positive
  }

    @Positive
  })
    @Positive
  public boolean client4() {
    @Positive
    return withcondpostconditionfunc2();
    @Positive
  }

  // :: error: (contracts.postcondition)
    @Positive
  public void withpostconditionsfunc1() {
    @Positive
    v1 = value1.length() - 3; // condition not satisfied here
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
  }

    @Positive
  public boolean withcondpostconditionsfunc2() {
    @Positive
    v1 = value1.length() - 3; // condition not satisfied here
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    @Positive
    return true;
    @Positive
  }
