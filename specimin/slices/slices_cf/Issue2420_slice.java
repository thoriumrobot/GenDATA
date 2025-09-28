  static void str(String argStr) {
    if (argStr.isEmpty()) {
      return;
    }
    if (argStr == "abc") {
      return;
    }
    // :: error: (argument)
    char c = "abc".charAt(argStr.length() - 1);
    // :: error: (argument)
    char c2 = "abc".charAt(argStr.length());
  }
