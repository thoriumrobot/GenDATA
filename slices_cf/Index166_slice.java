  public void testMethodInvocation() {
    requiresIndex("012345", 5);
    // :: error: (argument)
    requiresIndex("012345", 6);
  }
