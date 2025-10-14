/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_numeric_literal

public class Demo {
  public int sum(int[] a){
    int s = 0;
    for(int i=0;i<a.length;i++){
      s = s + a[i];
    }
    return s;
  }
  public String choose(boolean b, String x, String y){
    if(b){
      return x + "!";
    } else {
      return y + "?";
    }
  }
}
