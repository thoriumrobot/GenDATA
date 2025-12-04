/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_variable_operation

public class Sample {\n  public static int sum(int[] a){\n    int s = 0;\n    for (int i=0;i<a.length;i++) s += a[i];\n    return s;\n  }\n  public static void main(String[] args){\n    int[] arr = new int[]{1,2,3};\n    System.out.println(sum(arr));\n  }\n}\n