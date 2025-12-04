    @Positive
  public void testMethodInvocation() {
    @Positive
    requiresIndex("012345", 5);
    // :: error: (argument)
    @Positive
    requiresIndex("012345", 6);
    @Positive
  }
