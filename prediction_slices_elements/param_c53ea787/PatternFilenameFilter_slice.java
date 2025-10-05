// Source-based slice around line 71
// Method: <com.google.common.io.PatternFilenameFilter: boolean accept(File,String)>

   * someone still manages to pass null, let's continue to have the method work.
   *
   * (PatternFilenameFilter is of course one of those classes that shouldn't be a publicly visible
   * class to begin with but rather something returned from a static factory method whose declared
   * return type is plain FilenameFilter. If we made such a change, then the annotation we choose
   * here would have no significance to end users, who would be forced to conform to the signature
   * used in FilenameFilter.)
   */
  @Override
  public boolean accept(File dir, String fileName) {
    return pattern.matcher(fileName).matches();
  }
}
