#!/usr/bin/env python3
import os
import tempfile
import subprocess

from jdt_semantic_transformer import JdtSemanticTransformer


JAVA_HEADER = """
public class T {
    %s
}
""".strip()


def _write_java(method_src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.java')
    os.close(fd)
    with open(path, 'w') as f:
        f.write(JAVA_HEADER % method_src)
    return path


def _transform(java_path: str, transformations):
    out_path = java_path + '.out.java'
    with open(java_path, 'r') as f:
        original = f.read()
    tr = JdtSemanticTransformer()
    assert tr.transform_file(java_path, out_path, transformations, mode='enhanced')
    with open(out_path, 'r') as f:
        transformed = f.read()
    return original, transformed


def _javac(java_path: str) -> bool:
    res = subprocess.run(['javac', java_path], capture_output=True, text=True)
    return res.returncode == 0


def test_loop_conversion():
    src = "void m(){ for(int i=0;i<3;i++){ System.out.println(i); } }"
    java = _write_java(src)
    orig, out = _transform(java, ['loop_conversion'])
    assert orig != out
    assert 'for(int i=0;i<3;i++)' not in out
    assert 'while' in out
    assert _javac(java)


def test_guard_reversal():
    src = "void m(){ if(a){System.out.println(1);} else {System.out.println(2);} }"
    java = _write_java(src)
    orig, out = _transform(java, ['guard_reversal'])
    assert orig != out
    assert ('if (!' in out) or ('if(!' in out)


def test_mathematical_expression():
    src = "int m(){ int a = 1 + 2; return a; }"
    java = _write_java(src)
    orig, out = _transform(java, ['mathematical_expression'])
    assert orig != out


def test_logical_expression():
    src = "boolean m(boolean a, boolean b){ return a && b; }"
    java = _write_java(src)
    orig, out = _transform(java, ['logical_expression'])
    assert orig != out
    assert ('||' in out) or ('!' in out)


def test_ternary_operator():
    src = "int m(boolean c,int x,int y){ return c?x:y; }"
    java = _write_java(src)
    orig, out = _transform(java, ['ternary_operator'])
    assert orig != out
    assert 'if' in out


def test_switch_statement():
    src = "void m(int x){ switch(x){ case 1: System.out.println(1); break; default: break; } }"
    java = _write_java(src)
    orig, out = _transform(java, ['switch_statement'])
    assert orig != out
    assert 'if' in out


def test_variable_operation():
    src = "void m(){ int x=0; x+=1; }"
    java = _write_java(src)
    orig, out = _transform(java, ['variable_operation'])
    assert orig != out
    assert '+=' not in out


def test_method_extraction():
    src = "void m(){ int x=0; x++; }"
    java = _write_java(src)
    orig, out = _transform(java, ['method_extraction'])
    assert orig != out


def test_conditional_expression():
    src = "int m(boolean c,int x,int y){ return (c?x:y); }"
    java = _write_java(src)
    orig, out = _transform(java, ['conditional_expression'])
    assert orig != out


def test_array_access_pattern():
    src = "int m(int[] a){ return a[0]; }"
    java = _write_java(src)
    orig, out = _transform(java, ['array_access_pattern'])
    assert orig != out
    assert 'a[0]' not in out


def test_string_concatenation():
    src = "String m(){ return \"a\"+\"b\"; }"
    java = _write_java(src)
    orig, out = _transform(java, ['string_concatenation'])
    assert orig != out
    assert 'String.valueOf' in out


def test_numeric_literal():
    src = "int m(){ return 1000; }"
    java = _write_java(src)
    orig, out = _transform(java, ['numeric_literal'])
    assert orig != out
    assert '1_000' in out


def test_exception_handling():
    src = "void m(){ try{ int a=0; } catch(Exception e){} }"
    java = _write_java(src)
    orig, out = _transform(java, ['exception_handling'])
    assert orig != out
    assert 'finally' in out


def test_lambda_expression():
    src = "void m(){ java.util.function.Function<Integer,Integer> f = x->x; }"
    java = _write_java(src)
    orig, out = _transform(java, ['lambda_expression'])
    assert orig != out
    assert 'return' in out


def test_stream_api():
    src = "void m(){ java.util.List<Integer> xs = new java.util.ArrayList<>(); xs.stream().forEach(System.out::println); }"
    java = _write_java(src)
    orig, out = _transform(java, ['stream_api'])
    assert orig != out
    assert 'for (' in out or '->' in out


def test_builder_pattern():
    src = "void m(){ String s = new StringBuilder().append(\"a\").append(\"b\").toString(); }"
    java = _write_java(src)
    orig, out = _transform(java, ['builder_pattern'])
    assert orig != out


def test_functional_conversion():
    src = "void m(){ java.util.function.Function<String,String> f = x->x.toString(); }"
    java = _write_java(src)
    orig, out = _transform(java, ['functional_conversion'])
    assert orig != out


def test_simple_method_call():
    src = "void m(){ System.out.println(1); }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_method_call'])
    assert orig != out


def test_simple_assignment():
    src = "void m(){ int a; a = 1; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_assignment'])
    assert orig != out


def test_simple_conditional():
    src = "void m(){ if(true) System.out.println(1); }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_conditional'])
    assert orig != out


def test_simple_array_access():
    src = "int m(int[] a){ return a[1]; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_array_access'])
    assert orig != out


def test_simple_return_statement():
    src = "int m(){ return 1; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_return_statement'])
    assert orig != out


def test_simple_variable_declaration():
    src = "void m(){ int x = 1; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_variable_declaration'])
    assert orig != out


def test_simple_constructor_call():
    src = "void m(){ String s = new String(\"a\"); }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_constructor_call'])
    assert orig != out


def test_simple_field_access():
    src = "int m(){ this.hashCode(); return this.hashCode(); }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_field_access'])
    assert orig != out


def test_simple_string_operation():
    src = "String m(){ return \"a\"+\"b\"; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_string_operation'])
    assert orig != out


def test_simple_numeric_operation():
    src = "int m(){ return 1+2; }"
    java = _write_java(src)
    orig, out = _transform(java, ['simple_numeric_operation'])
    assert orig != out


def test_random_method_insertion():
    src = "void m(){ int a=0; }"
    java = _write_java(src)
    orig, out = _transform(java, ['random_method_insertion'])
    assert orig != out


def test_random_statement_insertion():
    src = "void m(){ int a=0; a++; }"
    java = _write_java(src)
    orig, out = _transform(java, ['random_statement_insertion'])
    assert orig != out


def test_random_expression_insertion():
    src = "int m(){ return 2; }"
    java = _write_java(src)
    orig, out = _transform(java, ['random_expression_insertion'])
    assert orig != out


